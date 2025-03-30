/*

Copyright (C) 2024 Matthew Brooks (Persist to and restore from disk)
Copyright (C) 2024 Matthew Brooks (Lua integration, train/evaluate, active clause, autoencoder, regressor)
Copyright (C) 2024 Matthew Brooks (Loss scaling, multi-threading, auto-vectorizer support)
Copyright (C) 2019 Ole-Christoffer Granmo (Original classifier C implementation)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include "lua.h"
#include "lauxlib.h"

#include <assert.h>
#include <errno.h>
#include <lauxlib.h>
#include <limits.h>
#include <lua.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define TK_TSETLIN_MT "santoku_tsetlin"

#define BITS (sizeof(unsigned int) * CHAR_BIT)

typedef enum {
  TM_CLASSIFIER,
  TM_ENCODER,
  TM_AUTO_ENCODER,
  TM_REGRESSOR,
} tsetlin_type_t;

typedef struct {
  unsigned int classes;
  unsigned int features;
  unsigned int clauses;
  unsigned int threshold;
  unsigned int state_bits;
  bool boost_true_positive;
  unsigned int input_bits;
  unsigned int input_chunks;
  unsigned int clause_chunks;
  unsigned int state_chunks;
  unsigned int action_chunks;
  unsigned int filter;
  unsigned int *state; // class clause bit chunk
  unsigned int *actions; // class clause chunk
  unsigned int *active_clause; // class clause chunk
  double specificity_low;
  double specificity_high;
  double *specificity;
  pthread_mutex_t *locks;
} tsetlin_classifier_t;

typedef struct {
  unsigned int encoding_bits;
  unsigned int encoding_chunks;
  unsigned int encoding_filter;
  tsetlin_classifier_t encoder;
} tsetlin_encoder_t;

typedef struct {
  tsetlin_encoder_t encoder;
  tsetlin_encoder_t decoder;
} tsetlin_auto_encoder_t;

typedef struct {
  tsetlin_classifier_t classifier;
} tsetlin_regressor_t;

typedef struct {
  tsetlin_type_t type;
  bool has_state;
  union {
    tsetlin_classifier_t *classifier;
    tsetlin_encoder_t *encoder;
    tsetlin_auto_encoder_t *auto_encoder;
    tsetlin_regressor_t *regressor;
  };
} tsetlin_t;

#define tm_state_get_lock(tm, class, clause, chunk) \
  (&(tm)->locks[(class) * (tm)->clauses * (tm)->input_chunks + \
                (clause) * (tm)->input_chunks + \
                (chunk)])

#define tm_specificity(tm, j) ((tm)->specificity[j])

#define tm_state_lock(tm, class, clause, chunk) \
	pthread_mutex_lock(tm_state_get_lock(tm, class, clause, chunk));

#define tm_state_unlock(tm, class, clause, chunk) \
	pthread_mutex_unlock(tm_state_get_lock(tm, class, clause, chunk));

#define tm_state_counts(tm, class, clause, input_chunk) \
  (&(tm)->state[(class) * (tm)->clauses * (tm)->input_chunks * ((tm)->state_bits - 1) + \
                (clause) * (tm)->input_chunks * ((tm)->state_bits - 1) + \
                (input_chunk) * ((tm)->state_bits - 1)])

#define tm_state_actions(tm, class, clause) \
  (&(tm)->actions[(class) * (tm)->clauses * (tm)->input_chunks + \
                  (clause) * (tm)->input_chunks])

static uint64_t const multiplier = 6364136223846793005u;
__thread uint64_t mcg_state = 0xcafef00dd15ea5e5u;

static inline uint32_t fast_rand ()
{
  uint64_t x = mcg_state;
  unsigned int count = (unsigned int) (x >> 61);
  mcg_state = x * multiplier;
  return (uint32_t) ((x ^ x >> 22) >> (22 + count));
}

static inline void seed_rand ()
{
  mcg_state = (uint64_t) pthread_self() ^ (uint64_t) time(NULL);
}

static inline double fast_drand ()
{
  return ((double)fast_rand()) / ((double)UINT32_MAX);
}

static inline bool fast_chance (double p)
{
  return fast_drand() <= p;
}

static inline int normal (double mean, double variance)
{
  double u1 = (double) (fast_rand() + 1) / ((double) UINT32_MAX + 1);
  double u2 = (double) fast_rand() / UINT32_MAX;
  double n1 = sqrt(-2 * log(u1)) * sin(8 * atan(1) * u2);
  return (int) round(mean + sqrt(variance) * n1);
}

static inline unsigned int popcount (
  unsigned int x
) {
  x = x - ((x >> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  x = (x + (x >> 4)) & 0x0F0F0F0F;
  x = x + (x >> 8);
  x = x + (x >> 16);
  return x & 0x0000003F;
}

static inline void init_shuffle (
  unsigned int *shuffle,
  unsigned int n
) {
  // TODO: Can we do this without two loop? Initialize and shuffle at the same
  // time?
  for (unsigned int i = 0; i < n; i ++)
    shuffle[i] = i;
  for (unsigned i = 0; i < n - 1; i ++) {
    unsigned int j = i + fast_rand() / (UINT32_MAX / (n - i) + 1);
    unsigned int t = shuffle[j];
    shuffle[j] = shuffle[i];
    shuffle[i] = t;
  }
}

static inline unsigned int hamming (
  unsigned int *a,
  unsigned int *b,
  unsigned int bits
) {
  unsigned int chunks = (bits - 1) / BITS + 1;
  unsigned int distance = 0;
  for (unsigned int i = 0; i < chunks; i ++) {
    unsigned int diff = a[i] ^ b[i];
    distance += (unsigned int) popcount(diff);
  }
  return distance;
}

static inline double hamming_loss (
  unsigned int *a,
  unsigned int *b,
  unsigned int bits,
  double alpha
) {
  double loss = hamming(a, b, bits);
  return pow(loss / (double) bits, alpha);
}

static inline double hamming_to_loss (
  unsigned int diff,
  unsigned int bits,
  double alpha
) {
  return pow((double) diff / (double) bits, alpha);
}

static inline double triplet_loss_hamming (
  unsigned int *a,
  unsigned int *n,
  unsigned int *p,
  unsigned int bits,
  double margin,
  double alpha
) {
  double dist_an = (double) hamming(a, n, bits) / (double) bits;
  double dist_ap = (double) hamming(a, p, bits) / (double) bits;
  return pow(fmin(1.0, fmax(0.0, (double) dist_ap - dist_an + margin)), alpha);
}

static inline void flip_bits (
  unsigned int *a,
  unsigned int n
) {
  for (unsigned int i = 0; i < n; i ++)
    a[i] = ~a[i];
}

static inline void tk_lua_callmod (
  lua_State *L,
  int nargs,
  int nret,
  const char *smod,
  const char *sfn
) {
  lua_getglobal(L, "require"); // arg req
  lua_pushstring(L, smod); // arg req smod
  lua_call(L, 1, 1); // arg mod
  lua_pushstring(L, sfn); // args mod sfn
  lua_gettable(L, -2); // args mod fn
  lua_remove(L, -2); // args fn
  lua_insert(L, - nargs - 1); // fn args
  lua_call(L, nargs, nret); // results
}

static inline int tk_lua_error (lua_State *L, const char *err)
{
  lua_pushstring(L, err);
  tk_lua_callmod(L, 1, 0, "santoku.error", "error");
  return 0;
}

// TODO: include the field name in error
static inline lua_Integer tk_lua_ftype (lua_State *L, int i, char *field)
{
  lua_getfield(L, i, field);
  int t = lua_type(L, -1);
  lua_pop(L, 1);
  return t;
}

static inline int tk_error (
  lua_State *L,
  const char *label,
  int err
) {
  lua_pushstring(L, label);
  lua_pushstring(L, strerror(err));
  tk_lua_callmod(L, 2, 0, "santoku.error", "error");
  return 1;
}

static inline const char *tk_lua_fcheckstring (lua_State *L, int i, char *field)
{
  lua_getfield(L, i, field);
  const char *s = luaL_checkstring(L, -1);
  lua_pop(L, 1);
  return s;
}

static inline double tk_lua_checkposdouble (lua_State *L, int i)
{
  lua_Number l = luaL_checknumber(L, i);
  if (l < 0)
    luaL_error(L, "value can't be negative");
  return (double) l;
}

static inline double tk_lua_fcheckposdouble (lua_State *L, int i, char *field)
{
  lua_getfield(L, i, field);
  double n = tk_lua_checkposdouble(L, -1);
  lua_pop(L, 1);
  return n;
}

static inline unsigned int tk_lua_checkunsigned (lua_State *L, int i)
{
  lua_Integer l = luaL_checkinteger(L, i);
  if (l < 0)
    luaL_error(L, "value can't be negative");
  if (l > UINT_MAX)
    luaL_error(L, "value is too large");
  return (unsigned int) l;
}

static inline lua_Integer tk_lua_fcheckunsigned (lua_State *L, int i, char *field)
{
  lua_getfield(L, i, field);
  lua_Integer n = tk_lua_checkunsigned(L, -1);
  lua_pop(L, 1);
  return n;
}

static inline bool tk_lua_fcheckboolean (lua_State *L, int i, char *field)
{
  lua_getfield(L, i, field);
  luaL_checktype(L, -1, LUA_TBOOLEAN);
  bool n = lua_toboolean(L, -1);
  lua_pop(L, 1);
  return n;
}

static inline bool tk_lua_optboolean (lua_State *L, int i, bool def)
{
  if (lua_type(L, i) == LUA_TNIL)
    return def;
  luaL_checktype(L, i, LUA_TBOOLEAN);
  return lua_toboolean(L, i);
}

static inline bool tk_lua_foptboolean (lua_State *L, int i, char *field, bool def)
{
  lua_getfield(L, i, field);
  bool b = tk_lua_optboolean(L, -1, def);
  lua_pop(L, 1);
  return b;
}

static inline int tk_lua_errmalloc (lua_State *L)
{
  lua_pushstring(L, "Error in malloc");
  tk_lua_callmod(L, 1, 0, "santoku.error", "error");
  return 0;
}

static inline int tk_lua_errno (lua_State *L, int err)
{
  lua_pushstring(L, strerror(errno));
  lua_pushinteger(L, err);
  tk_lua_callmod(L, 2, 0, "santoku.error", "error");
  return 0;
}

static inline FILE *tk_lua_tmpfile (lua_State *L)
{
  FILE *fh = tmpfile();
  if (fh) return fh;
  int e = errno;
  lua_settop(L, 0);
  lua_pushstring(L, "Error opening tmpfile");
  lua_pushstring(L, strerror(e));
  lua_pushinteger(L, e);
  tk_lua_callmod(L, 3, 0, "santoku.error", "error");
  return NULL;
}

static inline FILE *tk_lua_fmemopen (lua_State *L, char *data, size_t size, const char *flag)
{
  FILE *fh = fmemopen(data, size, flag);
  if (fh) return fh;
  int e = errno;
  lua_settop(L, 0);
  lua_pushstring(L, "Error opening string as file");
  lua_pushstring(L, strerror(e));
  lua_pushinteger(L, e);
  tk_lua_callmod(L, 3, 0, "santoku.error", "error");
  return NULL;
}

static inline FILE *tk_lua_fopen (lua_State *L, const char *fp, const char *flag)
{
  FILE *fh = fopen(fp, flag);
  if (fh) return fh;
  int e = errno;
  lua_settop(L, 0);
  lua_pushstring(L, "Error opening file");
  lua_pushstring(L, fp);
  lua_pushstring(L, strerror(e));
  lua_pushinteger(L, e);
  tk_lua_callmod(L, 4, 0, "santoku.error", "error");
  return NULL;
}

static inline void tk_lua_fclose (lua_State *L, FILE *fh)
{
  if (!fclose(fh)) return;
  int e = errno;
  lua_settop(L, 0);
  lua_pushstring(L, "Error closing file");
  lua_pushstring(L, strerror(e));
  lua_pushinteger(L, e);
  tk_lua_callmod(L, 3, 0, "santoku.error", "error");
}

static inline void tk_lua_fwrite (lua_State *L, void *data, size_t size, size_t memb, FILE *fh)
{
  fwrite(data, size, memb, fh);
  if (!ferror(fh)) return;
  int e = errno;
  lua_settop(L, 0);
  lua_pushstring(L, "Error writing to file");
  lua_pushstring(L, strerror(e));
  lua_pushinteger(L, e);
  tk_lua_callmod(L, 3, 0, "santoku.error", "error");
}

static inline void tk_lua_fread (lua_State *L, void *data, size_t size, size_t memb, FILE *fh)
{
  size_t r = fread(data, size, memb, fh);
  if (!ferror(fh) || !r) return;
  int e = errno;
  lua_settop(L, 0);
  lua_pushstring(L, "Error reading from file");
  lua_pushstring(L, strerror(e));
  lua_pushinteger(L, e);
  tk_lua_callmod(L, 3, 0, "santoku.error", "error");
}

static inline void tk_lua_fseek (lua_State *L, size_t size, size_t memb, FILE *fh)
{
  int r = fseek(fh, (long) (size * memb), SEEK_CUR);
  if (!ferror(fh) || !r) return;
  int e = errno;
  lua_settop(L, 0);
  lua_pushstring(L, "Error reading from file");
  lua_pushstring(L, strerror(e));
  lua_pushinteger(L, e);
  tk_lua_callmod(L, 3, 0, "santoku.error", "error");
}

static inline char *tk_lua_fslurp (lua_State *L, FILE *fh, size_t *len)
{
  if (fseek(fh, 0, SEEK_END) != 0) {
    tk_lua_errno(L, errno);
    return NULL;
  }
  long size = ftell(fh);
  if (size < 0) {
    tk_lua_errno(L, errno);
    return NULL;
  }
  if (fseek(fh, 0, SEEK_SET) != 0) {
    tk_lua_errno(L, errno);
    return NULL;
  }
  char *buffer = malloc((size_t) size);
  if (!buffer) {
    tk_lua_errmalloc(L);
    return NULL;
  }
  if (fread(buffer, 1, (size_t) size, fh) != (size_t) size) {
    free(buffer);
    tk_lua_errno(L, errno);
    return NULL;
  }
  *len = (size_t) size;
  return buffer;
}

static inline void tm_initialize_random_streams (
  tsetlin_classifier_t *tm,
  unsigned int *feedback_to_la,
  double specificity
) {
  unsigned int features = tm->features;
  unsigned int input_chunks = tm->input_chunks;
  memset(feedback_to_la, 0, input_chunks * sizeof(unsigned int));
  long int n = 2 * features;
  double p = 1.0 / specificity;
  long int active = normal(n * p, n * p * (1 - p));
  active = active >= n ? n : active;
  active = active < 0 ? 0 : active;
  for (unsigned int i = 0; i < active; i ++) {
    unsigned int f = fast_rand() % (2 * features);
    while (feedback_to_la[f / BITS] & (1U << (f % BITS)))
      f = fast_rand() % (2 * features);
    feedback_to_la[f / BITS] |= 1U << (f % BITS);
  }
}

static inline void tm_inc (
  tsetlin_classifier_t *tm,
  unsigned int class,
  unsigned int clause,
  unsigned int chunk,
  unsigned int active
) {
  tm_state_lock(tm, class, clause, chunk);

  unsigned int m = tm->state_bits - 1;
  unsigned int *counts = tm_state_counts(tm, class, clause, chunk);
  unsigned int carry, carry_next;
  carry = active;
  for (unsigned int b = 0; b < m; b ++) {
    carry_next = counts[b] & carry;
    counts[b] ^= carry;
    carry = carry_next;
  }
  unsigned int *actions = tm_state_actions(tm, class, clause);
  carry_next = actions[chunk] & carry;
  actions[chunk] ^= carry;
  carry = carry_next;
  for (unsigned int b = 0; b < m; b ++)
    counts[b] |= carry;
  actions[chunk] |= carry;

  tm_state_unlock(tm, class, clause, chunk);
}

static inline void tm_dec (
  tsetlin_classifier_t *tm,
  unsigned int class,
  unsigned int clause,
  unsigned int chunk,
  unsigned int active
) {
  tm_state_lock(tm, class, clause, chunk);

  unsigned int m = tm->state_bits - 1;
  unsigned int *counts = tm_state_counts(tm, class, clause, chunk);
  unsigned int carry, carry_next;
  carry = active;
  for (unsigned int b = 0; b < m; b ++) {
    carry_next = (~counts[b]) & carry;
    counts[b] ^= carry;
    carry = carry_next;
  }
  unsigned int *actions = tm_state_actions(tm, class, clause);
  carry_next = (~actions[chunk]) & carry;
  actions[chunk] ^= carry;
  carry = carry_next;
  for (unsigned int b = 0; b < m; b ++)
    counts[b] &= ~carry;
  actions[chunk] &= ~carry;

  tm_state_unlock(tm, class, clause, chunk);
}

static inline long int sum_up_class_votes (
  tsetlin_classifier_t *tm,
  unsigned int *clause_output,
  bool predict
) {
  long int class_sum = 0;
  unsigned int *active = tm->active_clause;
  unsigned int clause_chunks = tm->clause_chunks;
  if (predict) {
    for (unsigned int i = 0; i < clause_chunks; i ++) {
      class_sum += popcount(clause_output[i] & 0x55555555); // 0101
      class_sum -= popcount(clause_output[i] & 0xaaaaaaaa); // 1010
    }
  } else {
    for (unsigned int i = 0; i < clause_chunks; i ++) {
      class_sum += popcount(clause_output[i] & 0x55555555 & active[i]); // 0101
      class_sum -= popcount(clause_output[i] & 0xaaaaaaaa & active[i]); // 1010
    }
  }
  long int threshold = tm->threshold;
  class_sum = (class_sum > threshold) ? threshold : class_sum;
  class_sum = (class_sum < -threshold) ? -threshold : class_sum;
  return class_sum;
}

static inline void tm_calculate_clause_output (
  tsetlin_classifier_t *tm,
  unsigned int class,
  unsigned int *input,
  unsigned int *clause_output,
  bool predict
) {
  memset(clause_output, 0, tm->clause_chunks * sizeof(unsigned int));
  unsigned int clauses = tm->clauses;
  unsigned int filter = tm->filter;
  for (unsigned int j = 0; j < clauses; j ++) {
    unsigned int output = 0;
    unsigned int all_exclude = 0;
    unsigned int input_chunks = tm->input_chunks;
    unsigned int clause_chunk = j / BITS;
    unsigned int clause_chunk_pos = j % BITS;
    unsigned int *actions = tm_state_actions(tm, class, j);
    for (unsigned int k = 0; k < input_chunks - 1; k ++) {
      output |= ((actions[k] & input[k]) ^ actions[k]);
      all_exclude |= actions[k];
    }
    output |=
      (actions[input_chunks - 1] & input[input_chunks - 1] & filter) ^
      (actions[input_chunks - 1] & filter);
    all_exclude |= ((actions[input_chunks - 1] & filter) ^ 0);
    output = !output && !(predict && !all_exclude);
    if (output)
      clause_output[clause_chunk] |= (1U << clause_chunk_pos);
  }
}

static inline void tm_update (
  tsetlin_classifier_t *tm,
  unsigned int class,
  unsigned int *input,
  unsigned int target,
  unsigned int *clause_output,
  unsigned int *feedback_to_clauses,
  unsigned int *feedback_to_la
) {
  tm_calculate_clause_output(tm, class, input, clause_output, false);
  long int class_sum = sum_up_class_votes(tm, clause_output, false);
  long int tgt = target ? 1 : 0;
  unsigned int input_chunks = tm->input_chunks;
  unsigned int clause_chunks = tm->clause_chunks;
  unsigned int boost_true_positive = tm->boost_true_positive;
  unsigned int clauses = tm->clauses;
  unsigned int threshold = tm->threshold;
  unsigned int *active_clause = tm->active_clause;
  float p = (1.0 / (threshold * 2)) * (threshold + (1 - 2 * tgt) * class_sum);
  memset(feedback_to_clauses, 0, clause_chunks * sizeof(unsigned int));
  for (unsigned int i = 0; i < clauses; i ++) {
    unsigned int clause_chunk = i / BITS;
    unsigned int clause_chunk_pos = i % BITS;
    feedback_to_clauses[clause_chunk] |= ((unsigned int) fast_chance(p) << clause_chunk_pos);
  }
  for (unsigned int i = 0; i < clause_chunks; i ++)
    feedback_to_clauses[i] &= active_clause[i];
  for (unsigned int j = 0; j < clauses; j ++) {
    long int jl = (long int) j;
    unsigned int clause_chunk = j / BITS;
    unsigned int clause_chunk_pos = j % BITS;
    unsigned int *actions = tm_state_actions(tm, class, j);
    double specificity = tm_specificity(tm, j);
    if (!(feedback_to_clauses[clause_chunk] & (1U << clause_chunk_pos)))
      continue;
    if ((2 * tgt - 1) * (1 - 2 * (jl & 1)) == -1) {
      // Type II feedback
      if ((clause_output[clause_chunk] & (1U << clause_chunk_pos)) > 0)
        for (unsigned int k = 0; k < input_chunks; k ++) {
          unsigned int active = (~input[k]) & (~actions[k]);
          tm_inc(tm, class, j, k, active);
        }
    } else if ((2 * tgt - 1) * (1 - 2 * (jl & 1)) == 1) {
      // Type I Feedback
      tm_initialize_random_streams(tm, feedback_to_la, specificity);
      if ((clause_output[clause_chunk] & (1U << clause_chunk_pos)) > 0) {
        if (boost_true_positive)
          for (unsigned int k = 0; k < input_chunks; k ++) {
            unsigned int chunk = input[k];
            tm_inc(tm, class, j, k, chunk);
          }
        else
          for (unsigned int k = 0; k < input_chunks; k ++) {
            unsigned int fb = input[k] & (~feedback_to_la[k]);
            tm_inc(tm, class, j, k, fb);
          }
        for (unsigned int k = 0; k < input_chunks; k ++) {
          unsigned int fb = (~input[k]) & feedback_to_la[k];
          tm_dec(tm, class, j, k, fb);
        }
      } else {
        for (unsigned int k = 0; k < input_chunks; k ++)
          tm_dec(tm, class, j, k, feedback_to_la[k]);
      }
    }
  }
}

static inline unsigned int tm_score_max (
  tsetlin_classifier_t *tm,
  long int *scores
) {
  unsigned int n_classes = tm->classes;
  unsigned int max_class = 0;
  long int max_score = scores[0];
  for (unsigned int class = 1; class < n_classes; class ++) {
    if (scores[class] > max_score) {
      max_class = class;
      max_score = scores[class];
    }
  }
  return max_class;
}

static inline void tm_score (
  tsetlin_classifier_t *tm,
  unsigned int *input,
  unsigned int *clause_output,
  long int *scores
) {
  unsigned int n_classes = tm->classes;
  for (unsigned int class = 0; class < n_classes; class ++) {
    tm_calculate_clause_output(tm, class, input, clause_output, true);
    scores[class] = sum_up_class_votes(tm, clause_output, true);
  }
}

static inline unsigned int mc_tm_predict (
  tsetlin_classifier_t *tm,
  unsigned int *input
) {
  unsigned int clause_output[tm->clause_chunks];
  long int scores[tm->classes];
  tm_score(tm, input, clause_output, scores);
  return tm_score_max(tm, scores);
}

static inline void mc_tm_update (
  tsetlin_classifier_t *tm,
  unsigned int *input,
  unsigned int class,
  unsigned int *clause_output,
  unsigned int *feedback_to_clauses,
  unsigned int *feedback_to_la
) {
  tm_update(tm, class, input, 1, clause_output, feedback_to_clauses, feedback_to_la);
  // TODO: Is there a faster way to do negative class selection? Is negative
  // class selection even worth it?
  unsigned int negative_class = (unsigned int) fast_rand() % tm->classes;
  while (negative_class == class)
    negative_class = (unsigned int) fast_rand() % tm->classes;
  tm_update(tm, negative_class, input, 0, clause_output, feedback_to_clauses, feedback_to_la);
}

static inline void ae_tm_decode (
  tsetlin_auto_encoder_t *tm,
  unsigned int *input,
  unsigned int *decoding,
  long int *scores
) {
  tsetlin_classifier_t *decoder = &tm->decoder.encoder;
  unsigned int decoder_classes = decoder->classes;
  unsigned int clause_output[decoder->clause_chunks];
  tm_score(decoder, input, clause_output, scores);

  for (unsigned int i = 0; i < decoder_classes; i ++)
  {
    unsigned int chunk = i / BITS;
    unsigned int pos = i % BITS;
    if (scores[i] > 0)
      decoding[chunk] |= (1U << pos);
    else
      decoding[chunk] &= ~(1U << pos);
  }
  decoding[tm->decoder.encoding_chunks - 1] &= decoder->filter;
}

static inline void ae_tm_encode (
  tsetlin_auto_encoder_t *tm,
  unsigned int *input,
  unsigned int *encoding,
  long int *scores
) {
  tsetlin_classifier_t *encoder = &tm->encoder.encoder;
  unsigned int encoder_classes = encoder->classes;
  unsigned int clause_output[encoder->clause_chunks];
  tm_score(encoder, input, clause_output, scores);

  for (unsigned int i = 0; i < encoder_classes; i ++)
  {
    unsigned int chunk = i / BITS;
    unsigned int pos = i % BITS;
    if (scores[i] > 0)
      encoding[chunk] |= (1U << pos);
    else
      encoding[chunk] &= ~(1U << pos);
  }
  encoding[tm->encoder.encoding_chunks - 1] &= encoder->filter;
}

static inline void en_tm_encode (
  tsetlin_encoder_t *tm,
  unsigned int *input,
  unsigned int *encoding,
  long int *scores
) {
  unsigned int encoder_classes = tm->encoder.classes;
  unsigned int clause_output[tm->encoder.clause_chunks];
  tm_score(&tm->encoder, input, clause_output, scores);
  for (unsigned int i = 0; i < encoder_classes; i ++)
  {
    unsigned int chunk = i / BITS;
    unsigned int pos = i % BITS;
    if (scores[i] > 0)
      encoding[chunk] |= (1U << pos);
    else
      encoding[chunk] &= ~(1U << pos);
  }
  encoding[tm->encoding_chunks - 1] &= tm->encoder.filter;
}

static inline void ae_tm_update (
  tsetlin_auto_encoder_t *tm,
  unsigned int *input,
  unsigned int *clause_output,
  unsigned int *feedback_to_clauses,
  unsigned int *feedback_to_la_e,
  unsigned int *feedback_to_la_d,
  long int *scores_e,
  long int *scores_d,
  double loss_alpha
) {
  unsigned int encoding[tm->encoder.encoding_chunks * 2]; // encoding + flipped bits for input to decoder
  unsigned int decoding[tm->decoder.encoding_chunks * 2]; // decoding + flipped bits to compare to input

  // encode input, copy, flip bits
  ae_tm_encode(tm, input, encoding, scores_e);
  memcpy(encoding + tm->encoder.encoding_chunks, encoding, tm->encoder.encoding_chunks * sizeof(unsigned int));
  flip_bits(encoding + tm->encoder.encoding_chunks, tm->encoder.encoding_chunks);

  // decode encoding, copy, flip bits
  ae_tm_decode(tm, encoding, decoding, scores_d);
  memcpy(decoding + tm->decoder.encoding_chunks, decoding, tm->decoder.encoding_chunks * sizeof(unsigned int));
  flip_bits(decoding + tm->decoder.encoding_chunks, tm->decoder.encoding_chunks);

  unsigned int diff = hamming(input, decoding, tm->encoder.encoder.input_bits);
  double loss = hamming_to_loss(diff, tm->encoder.encoder.input_bits, loss_alpha);

  if (fast_chance(1 - loss))
    return;

  for (unsigned int bit = 0; bit < tm->encoder.encoding_bits; bit ++) {
    unsigned int chunk0 = bit / BITS;
    unsigned int chunk1 = chunk0 + tm->encoder.encoding_chunks;
    unsigned int pos = bit % BITS;
    unsigned int bit_x = encoding[chunk0] & (1U << pos);
    unsigned int bit_x_flipped = !bit_x;
    encoding[chunk0] ^= (1U << pos);
    encoding[chunk1] ^= (1U << pos);
    ae_tm_decode(tm, encoding, decoding, scores_d);
    memcpy(decoding + tm->decoder.encoding_chunks, decoding, tm->decoder.encoding_chunks * sizeof(unsigned int));
    flip_bits(decoding + tm->decoder.encoding_chunks, tm->decoder.encoding_chunks);
    unsigned int diff0 = hamming(input, decoding, tm->encoder.encoder.input_bits);
    encoding[chunk0] ^= (1U << pos);
    encoding[chunk1] ^= (1U << pos);
    if (diff0 <= diff)
      tm_update(&tm->encoder.encoder, bit, input, bit_x_flipped, clause_output, feedback_to_clauses, feedback_to_la_e);
    else if (diff0 > diff)
      tm_update(&tm->encoder.encoder, bit, input, bit_x, clause_output, feedback_to_clauses, feedback_to_la_e);
  }

  for (unsigned int bit = 0; bit < tm->encoder.encoder.input_bits; bit ++) {
    unsigned int chunk = bit / BITS;
    unsigned int pos = bit % BITS;
    unsigned int bit_i = input[chunk] & (1U << pos);
    tm_update(&tm->decoder.encoder, bit, encoding, bit_i, clause_output, feedback_to_clauses, feedback_to_la_d);
  }
}

static inline void en_tm_update (
  tsetlin_encoder_t *tm,
  unsigned int *a,
  unsigned int *n,
  unsigned int *p,
  unsigned int *clause_output,
  unsigned int *feedback_to_clauses,
  unsigned int *feedback_to_la,
  long int *scores,
  double margin,
  double loss_alpha
) {

  tsetlin_classifier_t *encoder = &tm->encoder;
  unsigned int classes = encoder->classes;
  unsigned int encoding_chunks = tm->encoding_chunks;
  unsigned int encoding_bits = tm->encoding_bits;

  unsigned int encoding_a[encoding_chunks];
  unsigned int encoding_n[encoding_chunks];
  unsigned int encoding_p[encoding_chunks];

  en_tm_encode(tm, a, encoding_a, scores);
  en_tm_encode(tm, n, encoding_n, scores);
  en_tm_encode(tm, p, encoding_p, scores);

  double loss = triplet_loss_hamming(encoding_a, encoding_n, encoding_p, encoding_bits, margin, loss_alpha);

  if (fast_chance(1 - loss))
    return;

  for (unsigned int i = 0; i < classes; i ++) {
    unsigned int chunk = i / BITS;
    unsigned int pos = i % BITS;
    unsigned int bit_a = encoding_a[chunk] & (1U << pos);
    unsigned int bit_n = encoding_n[chunk] & (1U << pos);
    unsigned int bit_p = encoding_p[chunk] & (1U << pos);
    if ((bit_a && bit_n && bit_p) || (!bit_a && !bit_n && !bit_p)) {
      // flip n, keep a and p
      tm_update(encoder, i, a, bit_a, clause_output, feedback_to_clauses, feedback_to_la);
      tm_update(encoder, i, n, !bit_n, clause_output, feedback_to_clauses, feedback_to_la);
      tm_update(encoder, i, p, bit_p, clause_output, feedback_to_clauses, feedback_to_la);
    } else if ((bit_a && bit_n && !bit_p) || (!bit_a && !bit_n && bit_p)) {
      // flip a, keep n and p
      tm_update(encoder, i, a, !bit_a, clause_output, feedback_to_clauses, feedback_to_la);
      tm_update(encoder, i, n, bit_n, clause_output, feedback_to_clauses, feedback_to_la);
      tm_update(encoder, i, p, bit_p, clause_output, feedback_to_clauses, feedback_to_la);
    } else if ((bit_a && !bit_n && bit_p) || (!bit_a && bit_n && !bit_p)) {
      // keep all
      tm_update(encoder, i, a, bit_a, clause_output, feedback_to_clauses, feedback_to_la);
      tm_update(encoder, i, n, bit_n, clause_output, feedback_to_clauses, feedback_to_la);
      tm_update(encoder, i, p, bit_p, clause_output, feedback_to_clauses, feedback_to_la);
    } else if ((bit_a && !bit_n && !bit_p) || (!bit_a && bit_n && bit_p)) {
      // flip p, keep a and n
      tm_update(encoder, i, a, bit_a, clause_output, feedback_to_clauses, feedback_to_la);
      tm_update(encoder, i, n, bit_n, clause_output, feedback_to_clauses, feedback_to_la);
      tm_update(encoder, i, p, !bit_p, clause_output, feedback_to_clauses, feedback_to_la);
    }
  }

}

static inline void mc_tm_initialize_active_clause (
  tsetlin_classifier_t *tm,
  double active_clause
) {
  unsigned int clause_chunks = tm->clause_chunks;
  unsigned int *active_clauses = tm->active_clause;
  memset(active_clauses, 0xFF, clause_chunks * sizeof(unsigned int));
  for (unsigned int i = 0; i < clause_chunks; i ++)
    for (unsigned int j = 0; j < BITS; j ++)
      if (!fast_chance(active_clause))
        active_clauses[i] &= ~(1U << j);
}

tsetlin_t *tk_tsetlin_peek (lua_State *L, int i)
{
  return (tsetlin_t *) luaL_checkudata(L, i, TK_TSETLIN_MT);
}

static inline double tk_lua_optposdouble (lua_State *L, int i, double def)
{
  if (lua_type(L, i) < 1)
    return def;
  lua_Number l = luaL_checknumber(L, i);
  if (l < 0)
    luaL_error(L, "value can't be negative");
  return (double) l;
}

static inline bool tk_lua_checkboolean (lua_State *L, int i)
{
  if (lua_type(L, i) == LUA_TNIL)
    return false;
  luaL_checktype(L, i, LUA_TBOOLEAN);
  return lua_toboolean(L, i);
}

static inline void tk_lua_register (lua_State *L, luaL_Reg *regs, int nup)
{
  while (true) {
    if ((*regs).name == NULL)
      break;
    for (int i = 0; i < nup; i ++)
      lua_pushvalue(L, -nup); // t upsa upsb
    lua_pushcclosure(L, (*regs).func, nup); // t upsa fn
    lua_setfield(L, -nup - 2, (*regs).name); // t
    regs ++;
  }
  lua_pop(L, nup);
}

static inline tsetlin_t *tk_tsetlin_alloc (lua_State *L)
{
  tsetlin_t *tm = lua_newuserdata(L, sizeof(tsetlin_t)); // ud
  if (!tm) luaL_error(L, "error in malloc during creation");
  memset(tm, 0, sizeof(tsetlin_t));
  luaL_getmetatable(L, TK_TSETLIN_MT); // ud mt
  lua_setmetatable(L, -2); // ud
  return tm;
}

static inline tsetlin_t *tk_tsetlin_alloc_classifier (lua_State *L, bool has_state)
{
  tsetlin_t *tm = tk_tsetlin_alloc(L);
  tm->type = TM_CLASSIFIER;
  tm->has_state = has_state;
  tm->classifier = malloc(sizeof(tsetlin_classifier_t));
  if (!tm->classifier) luaL_error(L, "error in malloc during creation");
  memset(tm->classifier, 0, sizeof(tsetlin_classifier_t));
  return tm;
}

static inline tsetlin_t *tk_tsetlin_alloc_encoder (lua_State *L, bool has_state)
{
  tsetlin_t *tm = tk_tsetlin_alloc(L);
  tm->type = TM_ENCODER;
  tm->has_state = has_state;
  tm->encoder = malloc(sizeof(tsetlin_encoder_t));
  if (!tm->encoder) luaL_error(L, "error in malloc during creation");
  memset(tm->encoder, 0, sizeof(tsetlin_encoder_t));
  return tm;
}

static inline tsetlin_t *tk_tsetlin_alloc_auto_encoder (lua_State *L, bool has_state)
{
  tsetlin_t *tm = tk_tsetlin_alloc(L);
  tm->type = TM_AUTO_ENCODER;
  tm->has_state = has_state;
  tm->auto_encoder = malloc(sizeof(tsetlin_auto_encoder_t));
  if (!tm->auto_encoder) luaL_error(L, "error in malloc during creation");
  memset(tm->auto_encoder, 0, sizeof(tsetlin_auto_encoder_t));
  return tm;
}

static inline tsetlin_t *tk_tsetlin_alloc_regressor (lua_State *L, bool has_state)
{
  tsetlin_t *tm = tk_tsetlin_alloc(L);
  tm->type = TM_REGRESSOR;
  tm->has_state = has_state;
  tm->regressor = malloc(sizeof(tsetlin_regressor_t));
  if (!tm->regressor) luaL_error(L, "error in malloc during creation");
  memset(tm->regressor, 0, sizeof(tsetlin_regressor_t));
  return tm;
}

static inline void tk_tsetlin_init_classifier (
  lua_State *L,
  tsetlin_classifier_t *tm,
  unsigned int classes,
  unsigned int features,
  unsigned int clauses,
  unsigned int state_bits,
  unsigned int threshold,
  bool boost_true_positive,
  double specificity_low,
  double specificity_high
) {
  tm->classes = classes;
  tm->features = features;
  tm->clauses = clauses;
  tm->state_bits = state_bits;
  tm->threshold = threshold;
  tm->boost_true_positive = boost_true_positive;
  tm->input_bits = 2 * tm->features;
  tm->input_chunks = (tm->input_bits - 1) / BITS + 1;
  tm->clause_chunks = (tm->clauses - 1) / BITS + 1;
  tm->state_chunks = tm->classes * tm->clauses * (tm->state_bits - 1) * tm->input_chunks;
  tm->action_chunks = tm->classes * tm->clauses * tm->input_chunks;
  tm->filter = tm->input_bits % BITS != 0
    ? ~(((unsigned int) ~0) << (tm->input_bits % BITS))
    : (unsigned int) ~0;
  tm->state = malloc(sizeof(unsigned int) * tm->state_chunks);
  tm->actions = malloc(sizeof(unsigned int) * tm->action_chunks);
  tm->active_clause = malloc(sizeof(unsigned int) * tm->clause_chunks);
  tm->specificity_low = specificity_low;
  tm->specificity_high = specificity_high;
  tm->specificity = malloc(sizeof(double) * tm->clauses);
  for (unsigned int i = 0; i < tm->clauses; i ++)
    tm->specificity[i] = (1.0 * i / tm->clauses) * (tm->specificity_high - tm->specificity_low) + tm->specificity_low;
  tm->locks = malloc(sizeof(pthread_mutex_t) * tm->action_chunks);
  for (unsigned int i = 0; i < tm->action_chunks; i ++)
    pthread_mutex_init(&tm->locks[i], NULL);
  if (!(tm->active_clause && tm->state && tm->actions))
    luaL_error(L, "error in malloc during creation of classifier");
  for (unsigned int i = 0; i < tm->classes; i ++)
    for (unsigned int j = 0; j < tm->clauses; j ++)
      for (unsigned int k = 0; k < tm->input_chunks; k ++) {
        unsigned int m = tm->state_bits - 1;
        unsigned int *actions = tm_state_actions(tm, i, j);
        unsigned int *counts = tm_state_counts(tm, i, j, k);
        actions[k] = 0U;
        for (unsigned int b = 0; b < m; b ++)
          counts[b] = ~0U;
      }
}

static inline int tk_tsetlin_init_encoder (
  lua_State *L,
  tsetlin_encoder_t *tm,
  unsigned int encoding_bits,
  unsigned int features,
  unsigned int clauses,
  unsigned int state_bits,
  unsigned int threshold,
  bool boost_true_positive,
  double specificity_low,
  double specificity_high
) {
  tk_tsetlin_init_classifier(L, &tm->encoder,
      encoding_bits, features, clauses, state_bits, threshold,
      boost_true_positive, specificity_low, specificity_high);
  tm->encoding_bits = encoding_bits;
  tm->encoding_chunks = (encoding_bits - 1) / BITS + 1;
  tm->encoding_filter = encoding_bits % BITS != 0
    ? ~(((unsigned int) ~0) << (encoding_bits % BITS))
    : (unsigned int) ~0;
  return 0;
}

static inline void tk_tsetlin_create_classifier (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_alloc_classifier(L, true);
  lua_insert(L, 1);

  tk_tsetlin_init_classifier(L, tm->classifier,
      tk_lua_fcheckunsigned(L, 2, "classes"),
      tk_lua_fcheckunsigned(L, 2, "features"),
      tk_lua_fcheckunsigned(L, 2, "clauses"),
      tk_lua_fcheckunsigned(L, 2, "state_bits"),
      tk_lua_fcheckunsigned(L, 2, "target"),
      tk_lua_fcheckboolean(L, 2, "boost_true_positive"),
      tk_lua_fcheckposdouble(L, 2, "spec_low"),
      tk_lua_fcheckposdouble(L, 2, "spec_high"));

  lua_settop(L, 1);
}

static inline void tk_tsetlin_create_encoder (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_alloc_encoder(L, true);
  lua_insert(L, 1);

  tk_tsetlin_init_encoder(L, tm->encoder,
      tk_lua_fcheckunsigned(L, 2, "hidden"),
      tk_lua_fcheckunsigned(L, 2, "visible"),
      tk_lua_fcheckunsigned(L, 2, "clauses"),
      tk_lua_fcheckunsigned(L, 2, "state_bits"),
      tk_lua_fcheckunsigned(L, 2, "target"),
      tk_lua_fcheckboolean(L, 2, "boost_true_positive"),
      tk_lua_fcheckposdouble(L, 2, "spec_low"),
      tk_lua_fcheckposdouble(L, 2, "spec_high"));

  lua_settop(L, 1);
}

static inline void tk_tsetlin_create_auto_encoder (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_alloc_auto_encoder(L, true);
  lua_insert(L, 1);

  unsigned int encoding_bits = tk_lua_fcheckunsigned(L, 2, "hidden");
  unsigned int features = tk_lua_fcheckunsigned(L, 2, "visible");
  unsigned int clauses = tk_lua_fcheckunsigned(L, 2, "clauses");
  unsigned int state_bits = tk_lua_fcheckunsigned(L, 2, "state_bits");
  unsigned int threshold = tk_lua_fcheckunsigned(L, 2, "target");
  bool boost_true_positive = tk_lua_fcheckboolean(L, 2, "boost_true_positive");
  double specificity_low = tk_lua_fcheckposdouble(L, 2, "spec_low");
  double specificity_high = tk_lua_fcheckposdouble(L, 2, "spec_high");

  tk_tsetlin_init_encoder(L, &tm->auto_encoder->encoder,
      encoding_bits, features, clauses, state_bits, threshold, boost_true_positive, specificity_low, specificity_high);
  tk_tsetlin_init_encoder(L, &tm->auto_encoder->decoder,
      tm->auto_encoder->encoder.encoder.input_bits, encoding_bits, clauses, state_bits, threshold, boost_true_positive, specificity_low, specificity_high);

  lua_settop(L, 1);
}

static inline void tk_tsetlin_create_regressor (lua_State *L)
{
  luaL_error(L, "unimplemented: create regressor");
  lua_settop(L, 0);
}

// TODO: Instead of the user passing in a string name for the type, pass in the
// enum value (TM_CLASSIFIER, TM_ENCODER, etc.). Expose these enum values via
// the library table.
static inline int tk_tsetlin_create (lua_State *L)
{
  const char *type = luaL_checkstring(L, 1);
  if (!strcmp(type, "classifier")) {

    lua_remove(L, 1);
    tk_tsetlin_create_classifier(L);
    return 1;

  } else if (!strcmp(type, "encoder")) {

    lua_remove(L, 1);
    tk_tsetlin_create_encoder(L);
    return 1;

  } else if (!strcmp(type, "auto_encoder")) {

    lua_remove(L, 1);
    tk_tsetlin_create_auto_encoder(L);
    return 1;

  } else if (!strcmp(type, "regressor")) {

    lua_remove(L, 1);
    tk_tsetlin_create_regressor(L);
    return 1;

  } else {

    luaL_error(L, "unexpected tsetlin machine type in create");
    return 0;

  }
}

static inline void tk_tsetlin_destroy_classifier (tsetlin_classifier_t *tm)
{
  if (tm == NULL)
    return;
  free(tm->state);
  tm->state = NULL;
  free(tm->actions);
  tm->actions = NULL;
  free(tm->active_clause);
  tm->active_clause = NULL;
  if (tm->locks)
    for (unsigned int i = 0; i < tm->action_chunks; i ++)
      pthread_mutex_destroy(&tm->locks[i]);
  free(tm->locks);
  tm->locks = NULL;
}

static inline int tk_tsetlin_destroy (lua_State *L)
{
  lua_settop(L, 1);
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  switch (tm->type) {
    case TM_CLASSIFIER:
      if (tm->classifier) {
        tk_tsetlin_destroy_classifier(tm->classifier);
        free(tm->classifier);
        tm->classifier = NULL;
      }
      break;
    case TM_ENCODER:
      if (tm->encoder) {
        tk_tsetlin_destroy_classifier(&tm->encoder->encoder);
        free(tm->encoder);
        tm->encoder = NULL;
      }
      break;
    case TM_AUTO_ENCODER:
      if (tm->auto_encoder) {
        tk_tsetlin_destroy_classifier(&tm->auto_encoder->encoder.encoder);
        tk_tsetlin_destroy_classifier(&tm->auto_encoder->decoder.encoder);
        free(tm->auto_encoder);
        tm->auto_encoder = NULL;
      }
      break;
    case TM_REGRESSOR:
      luaL_error(L, "unimplemented: destroy regressor");
      break;
    default:
      return luaL_error(L, "unexpected tsetlin machine type in destroy");
  }
  return 0;
}

static inline int tk_tsetlin_predict_classifier (lua_State *L, tsetlin_classifier_t *tm)
{
  lua_settop(L, 2);
  unsigned int *bm = (unsigned int *) luaL_checkstring(L, 2);
  lua_pushinteger(L, mc_tm_predict(tm, bm));
  return 1;
}

static inline int tk_tsetlin_predict_encoder (lua_State *L, tsetlin_encoder_t *tm)
{
  lua_settop(L, 2);
  unsigned int *bm = (unsigned int *) luaL_checkstring(L, 2);
  unsigned int encoding_a[tm->encoding_chunks];
  long int scores[tm->encoder.classes];
  en_tm_encode(tm, bm, encoding_a, scores);
  lua_pushlstring(L, (char *) encoding_a, sizeof(unsigned int) * tm->encoding_chunks);
  lua_pushinteger(L, tm->encoder.classes);
  return 2;
}

static inline int tk_tsetlin_predict_auto_encoder (lua_State *L, tsetlin_auto_encoder_t *tm)
{
  lua_settop(L, 2);
  unsigned int *bm = (unsigned int *) luaL_checkstring(L, 2);
  unsigned int encoding_chunks = tm->encoder.encoding_chunks;
  unsigned int encoding[encoding_chunks];
  long int scores[tm->encoder.encoder.classes];
  ae_tm_encode(tm, bm, encoding, scores);
  lua_pushlstring(L, (char *) encoding, sizeof(unsigned int) * encoding_chunks);
  return 1;
}

static inline int tk_tsetlin_predict_regressor (lua_State *L, tsetlin_regressor_t *tm)
{
  luaL_error(L, "unimplemented: predict regressor");
  return 0;
}

static inline int tk_tsetlin_predict (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  switch (tm->type) {
    case TM_CLASSIFIER:
      return tk_tsetlin_predict_classifier(L, tm->classifier);
    case TM_ENCODER:
      return tk_tsetlin_predict_encoder(L, tm->encoder);
    case TM_AUTO_ENCODER:
      return tk_tsetlin_predict_auto_encoder(L, tm->auto_encoder);
    case TM_REGRESSOR:
      return tk_tsetlin_predict_regressor(L, tm->regressor);
    default:
      return luaL_error(L, "unexpected tsetlin machine type in predict");
  }
  return 0;
}

static inline int tk_tsetlin_update_classifier (lua_State *L, tsetlin_classifier_t *tm)
{
  unsigned int *bm = (unsigned int *) luaL_checkstring(L, 2);
  lua_Integer tgt = luaL_checkinteger(L, 3);
  if (tgt < 0)
    luaL_error(L, "target class must be greater than zero");
  double active_clause = tk_lua_checkposdouble(L, 4);
  mc_tm_initialize_active_clause(tm, active_clause);
  unsigned int clause_chunks = tm->clause_chunks;
  unsigned int input_chunks = tm->input_chunks;
  unsigned int clause_output[clause_chunks];
  unsigned int feedback_to_clauses[clause_chunks];
  unsigned int feedback_to_la[input_chunks];
  mc_tm_update(tm, bm, tgt, clause_output, feedback_to_clauses, feedback_to_la);
  return 0;
}

static inline int tk_tsetlin_update_encoder (
  lua_State *L,
  tsetlin_encoder_t *tm
) {
  unsigned int *a = (unsigned int *) luaL_checkstring(L, 2);
  unsigned int *n = (unsigned int *) luaL_checkstring(L, 3);
  unsigned int *p = (unsigned int *) luaL_checkstring(L, 4);
  double active_clause = tk_lua_checkposdouble(L, 5);
  double margin = tk_lua_checkposdouble(L, 6);
  double loss_alpha = tk_lua_checkposdouble(L, 7);
  mc_tm_initialize_active_clause(&tm->encoder, active_clause);
  unsigned int clause_output[tm->encoder.clause_chunks];
  unsigned int feedback_to_clauses[tm->encoder.clause_chunks];
  unsigned int feedback_to_la[tm->encoder.input_chunks];
  long int scores[tm->encoder.classes];
  en_tm_update(tm, a, n, p, clause_output, feedback_to_clauses, feedback_to_la, scores, margin, loss_alpha);
  return 0;
}

static inline int tk_tsetlin_update_auto_encoder (lua_State *L, tsetlin_auto_encoder_t *tm)
{
  unsigned int *bm = (unsigned int *) luaL_checkstring(L, 2);
  double active_clause = tk_lua_checkposdouble(L, 3);
  double loss_alpha = tk_lua_checkposdouble(L, 4);
  unsigned int clause_output[tm->encoder.encoder.clause_chunks];
  unsigned int feedback_to_clauses[tm->encoder.encoder.clause_chunks];
  unsigned int feedback_to_la_e[tm->encoder.encoder.input_chunks];
  unsigned int feedback_to_la_d[tm->encoder.encoding_chunks * 2];
  long int scores_e[tm->encoder.encoder.classes];
  long int scores_d[tm->decoder.encoder.classes];
  mc_tm_initialize_active_clause(&tm->encoder.encoder, active_clause);
  mc_tm_initialize_active_clause(&tm->decoder.encoder, active_clause);
  ae_tm_update(tm, bm,
      clause_output, feedback_to_clauses, feedback_to_la_e, feedback_to_la_d, scores_e, scores_d,
      loss_alpha);
  return 0;
}

static inline int tk_tsetlin_update_regressor (lua_State *L, tsetlin_regressor_t *tm)
{
  luaL_error(L, "unimplemented: update regressor");
  return 0;
}

static inline int tk_tsetlin_update (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  if (!tm->has_state)
    luaL_error(L, "can't update a model loaded without state");
  switch (tm->type) {
    case TM_CLASSIFIER:
      return tk_tsetlin_update_classifier(L, tm->classifier);
    case TM_ENCODER:
      return tk_tsetlin_update_encoder(L, tm->encoder);
    case TM_AUTO_ENCODER:
      return tk_tsetlin_update_auto_encoder(L, tm->auto_encoder);
    case TM_REGRESSOR:
      return tk_tsetlin_update_regressor(L, tm->regressor);
    default:
      return luaL_error(L, "unexpected tsetlin machine type in update");
  }
  return 0;
}

typedef struct {
  tsetlin_classifier_t *tm;
  unsigned int n;
  unsigned int *next;
  unsigned int *shuffle;
  unsigned int *ps;
  unsigned int *ss;
  pthread_mutex_t *qlock;
} train_classifier_thread_data_t;

static void *train_classifier_thread (void *arg)
{
  seed_rand();
  train_classifier_thread_data_t *data = (train_classifier_thread_data_t *) arg;
  unsigned int clause_chunks = data->tm->clause_chunks;
  unsigned int input_chunks = data->tm->input_chunks;
  unsigned int clause_output[clause_chunks];
  unsigned int feedback_to_clauses[clause_chunks];
  unsigned int feedback_to_la[input_chunks];
  while (1) {
    pthread_mutex_lock(data->qlock);
    unsigned int next = *data->next;
    (*data->next) += 1;
    pthread_mutex_unlock(data->qlock);
    if (next >= data->n)
      return NULL;
    unsigned int idx = data->shuffle[next];
    mc_tm_update(data->tm, data->ps + idx * input_chunks, data->ss[idx],
        clause_output, feedback_to_clauses, feedback_to_la);
  }
}

static inline unsigned int tk_tsetlin_get_nthreads (
  lua_State *L, int i, char *field
) {
  long ts;
  unsigned int n_threads;
  if (field == NULL) {
    if (lua_type(L, i) == LUA_TNIL)
      goto sysconf;
    n_threads = tk_lua_checkunsigned(L, 2);
  } else if (tk_lua_ftype(L, 2, field) != LUA_TNIL) {
    n_threads = tk_lua_fcheckunsigned(L, 2, field);
    goto check;
  } else {
    goto sysconf;
  }
check:
  if (!n_threads)
    return (unsigned int) tk_lua_error(L, "threads must be at least 1\n");
  return n_threads;
sysconf:
  ts = sysconf(_SC_NPROCESSORS_ONLN) - 1;
  if (ts <= 0)
    return (unsigned int) tk_error(L, "sysconf", errno);
  lua_pushinteger(L, ts);
  n_threads = tk_lua_checkunsigned(L, -1);
  lua_pop(L, 1);
  return n_threads;
}

static inline int tk_tsetlin_train_classifier (lua_State *L, tsetlin_classifier_t *tm)
{
  unsigned int n = tk_lua_fcheckunsigned(L, 2, "samples");
  unsigned int *ps = (unsigned int *) tk_lua_fcheckstring(L, 2, "problems");
  unsigned int *ss = (unsigned int *) tk_lua_fcheckstring(L, 2, "solutions");
  double active_clause = tk_lua_fcheckposdouble(L, 2, "active_clause");
  unsigned int n_threads = tk_tsetlin_get_nthreads(L, 2, "threads");
  mc_tm_initialize_active_clause(tm, active_clause);

  unsigned int next = 0;
  unsigned int shuffle[n];
  init_shuffle(shuffle, n);

  pthread_t threads[n_threads];
  pthread_mutex_t qlock;
  pthread_mutex_init(&qlock, NULL);
  train_classifier_thread_data_t thread_data[n_threads];

  for (unsigned int i = 0; i < n_threads; i++) {
    thread_data[i].tm = tm;
    thread_data[i].n = n;
    thread_data[i].next = &next;
    thread_data[i].shuffle = shuffle;
    thread_data[i].ps = ps;
    thread_data[i].ss = ss;
    thread_data[i].qlock = &qlock;
    if (pthread_create(&threads[i], NULL, train_classifier_thread, &thread_data[i]) != 0)
      return tk_error(L, "pthread_create", errno);
  }

  // TODO: Ensure these get freed on error above
  for (unsigned int i = 0; i < n_threads; i++)
    if (pthread_join(threads[i], NULL) != 0)
      return tk_error(L, "pthread_join", errno);

  pthread_mutex_destroy(&qlock);

  return 0;
}

typedef struct {
  tsetlin_encoder_t *tm;
  unsigned int n;
  unsigned int *next;
  unsigned int *shuffle;
  unsigned int *tokens;
  double margin;
  double loss_alpha;
  pthread_mutex_t *qlock;
} train_encoder_thread_data_t;

static void *train_encoder_thread (void *arg)
{
  seed_rand();
  train_encoder_thread_data_t *data = (train_encoder_thread_data_t *) arg;
  unsigned int clause_chunks = data->tm->encoder.clause_chunks;
  unsigned int encoder_classes = data->tm->encoder.classes;
  unsigned int input_chunks = data->tm->encoder.input_chunks;
  unsigned int clause_output[clause_chunks];
  unsigned int feedback_to_clauses[clause_chunks];
  unsigned int feedback_to_la[input_chunks];
  long int scores[encoder_classes];
  while (1) {
    pthread_mutex_lock(data->qlock);
    unsigned int next = *data->next;
    (*data->next) ++;
    pthread_mutex_unlock(data->qlock);
    if (next >= data->n)
      return NULL;
    unsigned int idx = data->shuffle[next];
    unsigned int *a = data->tokens + ((idx * 3 + 0) * input_chunks);
    unsigned int *n = data->tokens + ((idx * 3 + 1) * input_chunks);
    unsigned int *p = data->tokens + ((idx * 3 + 2) * input_chunks);
    en_tm_update(data->tm, a, n, p,
        clause_output, feedback_to_clauses, feedback_to_la, scores,
        data->margin, data->loss_alpha);
  }
  return NULL;
}

static inline int tk_tsetlin_train_encoder (
  lua_State *L,
  tsetlin_encoder_t *tm
) {
  unsigned int n = tk_lua_fcheckunsigned(L, 2, "samples");
  unsigned int *tokens = (unsigned int *) tk_lua_fcheckstring(L, 2, "corpus");
  double active_clause = tk_lua_fcheckposdouble(L, 2, "active_clause");
  double margin = tk_lua_fcheckposdouble(L, 2, "margin");
  double loss_alpha = tk_lua_fcheckposdouble(L, 2, "loss_alpha");
  unsigned int n_threads = tk_tsetlin_get_nthreads(L, 2, "threads");
  mc_tm_initialize_active_clause(&tm->encoder, active_clause);

  pthread_t threads[n_threads];
  pthread_mutex_t qlock;
  pthread_mutex_init(&qlock, NULL);
  train_encoder_thread_data_t thread_data[n_threads];

  unsigned int next = 0;
  unsigned int shuffle[n];
  init_shuffle(shuffle, n);

  for (unsigned int i = 0; i < n_threads; i++) {
    thread_data[i].tm = tm;
    thread_data[i].n = n;
    thread_data[i].next = &next;
    thread_data[i].shuffle = shuffle;
    thread_data[i].tokens = tokens;
    thread_data[i].margin = margin;
    thread_data[i].loss_alpha = loss_alpha;
    thread_data[i].qlock = &qlock;
    if (pthread_create(&threads[i], NULL, train_encoder_thread, &thread_data[i]) != 0)
      return tk_error(L, "pthread_create", errno);
  }

  // TODO: Ensure these get freed on error above
  for (unsigned int i = 0; i < n_threads; i++)
    if (pthread_join(threads[i], NULL) != 0)
      return tk_error(L, "pthread_join", errno);

  pthread_mutex_destroy(&qlock);

  return 0;
}

typedef struct {
  tsetlin_auto_encoder_t *tm;
  unsigned int n;
  unsigned int *next;
  unsigned int *shuffle;
  unsigned int *ps;
  double loss_alpha;
  pthread_mutex_t *qlock;
} train_auto_encoder_thread_data_t;

static void *train_auto_encoder_thread (void *arg)
{
  seed_rand();
  train_auto_encoder_thread_data_t *data = (train_auto_encoder_thread_data_t *) arg;
  unsigned int clause_output[data->tm->encoder.encoder.clause_chunks];
  unsigned int feedback_to_clauses[data->tm->encoder.encoder.clause_chunks];
  unsigned int feedback_to_la_e[data->tm->encoder.encoder.input_chunks];
  unsigned int feedback_to_la_d[data->tm->encoder.encoding_chunks * 2];
  long int scores_e[data->tm->encoder.encoder.classes];
  long int scores_d[data->tm->decoder.encoder.classes];
  while (1) {
    pthread_mutex_lock(data->qlock);
    unsigned int next = *data->next;
    (*data->next) += 1;
    pthread_mutex_unlock(data->qlock);
    if (next >= data->n)
      return NULL;
    unsigned int idx = data->shuffle[next];
    ae_tm_update(data->tm, data->ps + idx * data->tm->encoder.encoder.input_chunks,
        clause_output, feedback_to_clauses, feedback_to_la_e, feedback_to_la_d, scores_e, scores_d,
        data->loss_alpha);
  }
  return NULL;
}

static inline int tk_tsetlin_train_auto_encoder (lua_State *L, tsetlin_auto_encoder_t *tm)
{
  unsigned int n = tk_lua_fcheckunsigned(L, 2, "samples");
  unsigned int *ps = (unsigned int *) tk_lua_fcheckstring(L, 2, "corpus");
  double active_clause = tk_lua_fcheckposdouble(L, 2, "active_clause");
  double loss_alpha = tk_lua_fcheckposdouble(L, 2, "loss_alpha");
  unsigned int n_threads = tk_tsetlin_get_nthreads(L, 2, "threads");

  mc_tm_initialize_active_clause(&tm->encoder.encoder, active_clause);
  mc_tm_initialize_active_clause(&tm->decoder.encoder, active_clause);

  unsigned int next = 0;
  unsigned int shuffle[n];
  init_shuffle(shuffle, n);

  pthread_t threads[n_threads];
  pthread_mutex_t qlock;
  pthread_mutex_init(&qlock, NULL);
  train_auto_encoder_thread_data_t thread_data[n_threads];

  for (unsigned int i = 0; i < n_threads; i++) {
    thread_data[i].tm = tm;
    thread_data[i].n = n;
    thread_data[i].next = &next;
    thread_data[i].shuffle = shuffle;
    thread_data[i].ps = ps;
    thread_data[i].loss_alpha = loss_alpha;
    thread_data[i].qlock = &qlock;
    if (pthread_create(&threads[i], NULL, train_auto_encoder_thread, &thread_data[i]) != 0)
      return tk_error(L, "pthread_create", errno);
  }

  for (unsigned int i = 0; i < n_threads; i++)
    if (pthread_join(threads[i], NULL) != 0)
      return tk_error(L, "pthread_join", errno);

  return 0;
}

static inline int tk_tsetlin_train_regressor (lua_State *L, tsetlin_regressor_t *tm)
{
  luaL_error(L, "unimplemented: train regressor");
  return 0;
}

static inline int tk_tsetlin_train (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  if (!tm->has_state)
    luaL_error(L, "can't train a model loaded without state");
  switch (tm->type) {
    case TM_CLASSIFIER:
      return tk_tsetlin_train_classifier(L, tm->classifier);
    case TM_ENCODER:
      return tk_tsetlin_train_encoder(L, tm->encoder);
    case TM_AUTO_ENCODER:
      return tk_tsetlin_train_auto_encoder(L, tm->auto_encoder);
    case TM_REGRESSOR:
      return tk_tsetlin_train_regressor(L, tm->regressor);
    default:
      return luaL_error(L, "unexpected tsetlin machine type in train");
  }
}

typedef struct {
  tsetlin_classifier_t *tm;
  unsigned int n;
  unsigned int *next;
  unsigned int *ps;
  unsigned int *ss;
  bool track_stats;
  unsigned int *correct;
  unsigned int **observations;
  unsigned int **predictions;
  unsigned int **confusion;
  pthread_mutex_t *lock;
  pthread_mutex_t *qlock;
} evaluate_classifier_thread_data_t;

static void *evaluate_classifier_thread (void *arg)
{
  seed_rand();
  evaluate_classifier_thread_data_t *data = (evaluate_classifier_thread_data_t *) arg;
  unsigned int classes = data->tm->classes;
  unsigned int input_chunks = data->tm->input_chunks;
  while (1) {
    pthread_mutex_lock(data->qlock);
    unsigned int next = *data->next;
    (*data->next) += 1;
    pthread_mutex_unlock(data->qlock);
    if (next >= data->n)
      return NULL;
    unsigned int expected = data->ss[next];
    unsigned int predicted = mc_tm_predict(data->tm, data->ps + next * input_chunks);
    pthread_mutex_lock(data->lock);
    if (expected == predicted)
      (*data->correct) += 1;
    if (data->track_stats) {
      (*data->observations)[expected] ++;
      (*data->predictions)[predicted] ++;
      if (expected != predicted)
        (*data->confusion)[expected * classes + predicted] ++;
    }
    pthread_mutex_unlock(data->lock);
  }
}

static inline int tk_tsetlin_evaluate_classifier (
  lua_State *L,
  tsetlin_classifier_t *tm
) {

  lua_settop(L, 2);
  unsigned int n = tk_lua_fcheckunsigned(L, 2, "samples");
  unsigned int *ps = (unsigned int *) tk_lua_fcheckstring(L, 2, "problems");
  unsigned int *ss = (unsigned int *) tk_lua_fcheckstring(L, 2, "solutions");
  unsigned int n_threads = tk_tsetlin_get_nthreads(L, 2, "threads");
  bool track_stats = tk_lua_foptboolean(L, 2, "stats", false);
  unsigned int correct = 0;
  unsigned int *confusion = NULL;
  unsigned int *predictions = NULL;
  unsigned int *observations = NULL;
  unsigned int classes = tm->classes;

  if (track_stats) {
    confusion = malloc(sizeof(unsigned int) * classes * classes);
    predictions = malloc(sizeof(unsigned int) * classes);
    observations = malloc(sizeof(unsigned int) * classes);
    if (!(confusion && predictions && observations))
      luaL_error(L, "error in malloc during evaluation");
    memset(confusion, 0, sizeof(unsigned int) * classes * classes);
    memset(predictions, 0, sizeof(unsigned int) * classes);
    memset(observations, 0, sizeof(unsigned int) * classes);
  }

  pthread_t threads[n_threads];
  pthread_mutex_t lock;
  pthread_mutex_t qlock;
  pthread_mutex_init(&lock, NULL);
  pthread_mutex_init(&qlock, NULL);
  evaluate_classifier_thread_data_t thread_data[n_threads];

  unsigned int next = 0;

  for (unsigned int i = 0; i < n_threads; i++) {
    thread_data[i].tm = tm;
    thread_data[i].n = n;
    thread_data[i].next = &next;
    thread_data[i].ps = ps;
    thread_data[i].ss = ss;
    thread_data[i].track_stats = track_stats;
    thread_data[i].correct = &correct;
    thread_data[i].observations = &observations;
    thread_data[i].predictions = &predictions;
    thread_data[i].confusion = &confusion;
    thread_data[i].lock = &lock;
    thread_data[i].qlock = &qlock;
    if (pthread_create(&threads[i], NULL, evaluate_classifier_thread, &thread_data[i]) != 0)
      return tk_error(L, "pthread_create", errno);
  }

  // TODO: Ensure these get freed on error above
  for (unsigned int i = 0; i < n_threads; i++)
    if (pthread_join(threads[i], NULL) != 0)
      return tk_error(L, "pthread_join", errno);

  pthread_mutex_destroy(&lock);
  pthread_mutex_destroy(&qlock);

  lua_pushnumber(L, correct);
  if (track_stats) {
    lua_newtable(L); // ct
    for (unsigned int i = 0; i < classes; i ++) {
      for (unsigned int j = 0; j < classes; j ++) {
        unsigned int c = confusion[i * classes + j];
        if (c > 0) {
          lua_pushinteger(L, i); // ct i
          lua_gettable(L, -2); // ct t
          if (lua_type(L, -1) == LUA_TNIL) {
            lua_pop(L, 1); // ct
            lua_newtable(L); // ct t
            lua_pushinteger(L, i); // ct t i
            lua_pushvalue(L, -2); // ct t i t
            lua_settable(L, -4); // ct t
          }
          lua_pushinteger(L, j); // ct t j
          lua_pushinteger(L, c); // ct t j c
          lua_settable(L, -3); // ct t
          lua_pop(L, 1);
        }
      }
    }

    lua_newtable(L); // pt
    for (unsigned int i = 0; i < classes; i ++) {
      lua_pushinteger(L, i); // pt i
      lua_pushinteger(L, predictions[i]); // pt i p
      lua_settable(L, -3); // pt
    }

    lua_newtable(L); // ot
    for (unsigned int i = 0; i < classes; i ++) {
      lua_pushinteger(L, i); // ot i
      lua_pushinteger(L, observations[i]); // ot i o
      lua_settable(L, -3); // ot
    }

    free(confusion);
    free(predictions);
    free(observations);
    return 4;

  } else {
    return 1;
  }
}

typedef struct {
  tsetlin_encoder_t *tm;
  unsigned int n;
  unsigned int *next;
  unsigned int *tokens;
  double margin;
  unsigned int *correct;
  pthread_mutex_t *lock;
  pthread_mutex_t *qlock;
} evaluate_encoder_thread_data_t;

static void *evaluate_encoder_thread (void *arg)
{
  seed_rand();
  evaluate_encoder_thread_data_t *data = (evaluate_encoder_thread_data_t *) arg;
  unsigned int encoding_a[data->tm->encoding_chunks];
  unsigned int encoding_n[data->tm->encoding_chunks];
  unsigned int encoding_p[data->tm->encoding_chunks];
  long int scores[data->tm->encoder.classes];
  while (1) {
    pthread_mutex_lock(data->qlock);
    unsigned int next = *data->next;
    (*data->next) += 1;
    pthread_mutex_unlock(data->qlock);
    if (next >= data->n)
      return NULL;
    unsigned int *a = data->tokens + ((next * 3 + 0) * data->tm->encoder.input_chunks);
    unsigned int *n = data->tokens + ((next * 3 + 1) * data->tm->encoder.input_chunks);
    unsigned int *p = data->tokens + ((next * 3 + 2) * data->tm->encoder.input_chunks);
    en_tm_encode(data->tm, a, encoding_a, scores);
    en_tm_encode(data->tm, n, encoding_n, scores);
    en_tm_encode(data->tm, p, encoding_p, scores);
    unsigned int dist_an = hamming(encoding_a, encoding_n, data->tm->encoding_bits);
    unsigned int dist_ap = hamming(encoding_a, encoding_p, data->tm->encoding_bits);
    if (dist_ap < dist_an) {
      pthread_mutex_lock(data->lock);
      (*data->correct) += 1;
      pthread_mutex_unlock(data->lock);
    }
  }
  return NULL;
}

static inline int tk_tsetlin_evaluate_encoder (lua_State *L, tsetlin_encoder_t *tm)
{
  lua_settop(L, 2);
  unsigned int n = tk_lua_fcheckunsigned(L, 2, "samples");
  unsigned int *tokens = (unsigned int *) tk_lua_fcheckstring(L, 2, "corpus");
  unsigned int n_threads = tk_tsetlin_get_nthreads(L, 2, "threads");

  unsigned int correct = 0;

  pthread_t threads[n_threads];
  pthread_mutex_t lock;
  pthread_mutex_t qlock;
  pthread_mutex_init(&lock, NULL);
  pthread_mutex_init(&qlock, NULL);
  evaluate_encoder_thread_data_t thread_data[n_threads];

  unsigned int next = 0;

  for (unsigned int i = 0; i < n_threads; i++) {
    thread_data[i].tm = tm;
    thread_data[i].n = n;
    thread_data[i].next = &next;
    thread_data[i].tokens = tokens;
    thread_data[i].correct = &correct;
    thread_data[i].lock = &lock;
    thread_data[i].qlock = &qlock;
    if (pthread_create(&threads[i], NULL, evaluate_encoder_thread, &thread_data[i]) != 0)
      return tk_error(L, "pthread_create", errno);
  }

  // TODO: Ensure these get freed on error above
  for (unsigned int i = 0; i < n_threads; i++)
    if (pthread_join(threads[i], NULL) != 0)
      return tk_error(L, "pthread_join", errno);

  pthread_mutex_destroy(&lock);
  pthread_mutex_destroy(&qlock);

  lua_pushnumber(L, (double) correct / n);
  return 1;
}

typedef struct {
  tsetlin_auto_encoder_t *tm;
  unsigned int n;
  unsigned int *next;
  unsigned int *ps;
  unsigned int *total_diff;
  pthread_mutex_t *lock;
  pthread_mutex_t *qlock;
} evaluate_auto_encoder_thread_data_t;

static void *evaluate_auto_encoder_thread (void *arg)
{
  seed_rand();
  evaluate_auto_encoder_thread_data_t *data = (evaluate_auto_encoder_thread_data_t *) arg;
  unsigned int encoding[data->tm->encoder.encoding_chunks * 2];
  unsigned int decoding[data->tm->decoder.encoding_chunks * 2];
  long int scores_e[data->tm->encoder.encoder.classes];
  long int scores_d[data->tm->decoder.encoder.classes];
  while (1) {
    pthread_mutex_lock(data->qlock);
    unsigned int next = *data->next;
    (*data->next) += 1;
    pthread_mutex_unlock(data->qlock);
    if (next >= data->n)
      return NULL;
    unsigned int *input = data->ps + next * data->tm->encoder.encoder.input_chunks;
    ae_tm_encode(data->tm, input, encoding, scores_e);
    memcpy(encoding + data->tm->encoder.encoding_chunks, encoding, data->tm->encoder.encoding_chunks * sizeof(unsigned int));
    flip_bits(encoding + data->tm->encoder.encoding_chunks, data->tm->encoder.encoding_chunks);
    ae_tm_decode(data->tm, encoding, decoding, scores_d);
    memcpy(decoding + data->tm->decoder.encoding_chunks, decoding, data->tm->decoder.encoding_chunks * sizeof(unsigned int));
    flip_bits(decoding + data->tm->decoder.encoding_chunks, data->tm->decoder.encoding_chunks);
    pthread_mutex_lock(data->lock);
    (*data->total_diff) += hamming(input, decoding, data->tm->encoder.encoder.input_bits);
    pthread_mutex_unlock(data->lock);
  }
}

static inline int tk_tsetlin_evaluate_auto_encoder (lua_State *L, tsetlin_auto_encoder_t *tm)
{
  lua_settop(L, 2);
  unsigned int n = tk_lua_fcheckunsigned(L, 2, "samples");
  unsigned int *ps = (unsigned int *) tk_lua_fcheckstring(L, 2, "corpus");
  unsigned int n_threads = tk_tsetlin_get_nthreads(L, 2, "threads");
  unsigned int input_bits = tm->encoder.encoder.input_bits;

  unsigned int total_bits = n * input_bits;
  unsigned int total_diff = 0;

  pthread_t threads[n_threads];
  pthread_mutex_t lock;
  pthread_mutex_t qlock;
  pthread_mutex_init(&lock, NULL);
  pthread_mutex_init(&qlock, NULL);
  evaluate_auto_encoder_thread_data_t thread_data[n_threads];

  unsigned int next = 0;

  for (unsigned int i = 0; i < n_threads; i++) {
    thread_data[i].tm = tm;
    thread_data[i].n = n;
    thread_data[i].next = &next;
    thread_data[i].ps = ps;
    thread_data[i].total_diff = &total_diff;
    thread_data[i].lock = &lock;
    thread_data[i].qlock = &qlock;
    if (pthread_create(&threads[i], NULL, evaluate_auto_encoder_thread, &thread_data[i]) != 0)
      return tk_error(L, "pthread_create", errno);
  }

  // TODO: Ensure these get freed on error above
  for (unsigned int i = 0; i < n_threads; i++)
    if (pthread_join(threads[i], NULL) != 0)
      return tk_error(L, "pthread_join", errno);

  pthread_mutex_destroy(&lock);
  pthread_mutex_destroy(&qlock);

  lua_pushnumber(L, 1 - (double) total_diff / (double) total_bits);
  return 1;
}

static inline int tk_tsetlin_evaluate_regressor (lua_State *L, tsetlin_regressor_t *tm)
{
  luaL_error(L, "unimplemented: evaluate regressor");
  return 0;
}

static inline int tk_tsetlin_evaluate (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  switch (tm->type) {
    case TM_CLASSIFIER:
      return tk_tsetlin_evaluate_classifier(L, tm->classifier);
    case TM_ENCODER:
      return tk_tsetlin_evaluate_encoder(L, tm->encoder);
    case TM_AUTO_ENCODER:
      return tk_tsetlin_evaluate_auto_encoder(L, tm->auto_encoder);
    case TM_REGRESSOR:
      return tk_tsetlin_evaluate_regressor(L, tm->regressor);
    default:
      return luaL_error(L, "unexpected tsetlin machine type in evaluate");
  }
}

static inline void _tk_tsetlin_persist_classifier (lua_State *L, tsetlin_classifier_t *tm, FILE *fh, bool persist_state)
{
  tk_lua_fwrite(L, &tm->classes, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->features, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->clauses, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->threshold, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->state_bits, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->boost_true_positive, sizeof(bool), 1, fh);
  tk_lua_fwrite(L, &tm->input_bits, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->input_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->clause_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->state_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->action_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->filter, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->specificity_low, sizeof(double), 1, fh);
  tk_lua_fwrite(L, &tm->specificity_high, sizeof(double), 1, fh);
  if (persist_state)
    tk_lua_fwrite(L, tm->state, sizeof(unsigned int), tm->state_chunks, fh);
  tk_lua_fwrite(L, tm->actions, sizeof(unsigned int), tm->action_chunks, fh);
}

static inline void tk_tsetlin_persist_classifier (lua_State *L, tsetlin_classifier_t *tm, FILE *fh, bool persist_state)
{
  _tk_tsetlin_persist_classifier(L, tm, fh, persist_state);
}

static inline void tk_tsetlin_persist_encoder (lua_State *L, tsetlin_encoder_t *tm, FILE *fh, bool persist_state)
{
  tk_lua_fwrite(L, &tm->encoding_bits, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->encoding_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->encoding_filter, sizeof(unsigned int), 1, fh);
  _tk_tsetlin_persist_classifier(L, &tm->encoder, fh, persist_state);
}

static inline void tk_tsetlin_persist_auto_encoder (lua_State *L, tsetlin_auto_encoder_t *tm, FILE *fh, bool persist_state)
{
  tk_tsetlin_persist_encoder(L, &tm->encoder, fh, persist_state);
  tk_tsetlin_persist_encoder(L, &tm->decoder, fh, persist_state);
}

static inline void tk_tsetlin_persist_regressor (lua_State *L, tsetlin_regressor_t *tm, FILE *fh, bool persist_state)
{
  _tk_tsetlin_persist_classifier(L, &tm->classifier, fh, persist_state);
}

static inline int tk_tsetlin_persist (lua_State *L)
{
  lua_settop(L, 3);
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  bool tostr = lua_type(L, 2) == LUA_TNIL;
  FILE *fh = tostr ? tk_lua_tmpfile(L) : tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  bool persist_state = lua_toboolean(L, 3);
  if (persist_state && !tm->has_state)
    luaL_error(L, "can't persist the state of a model loaded without state");
  tk_lua_fwrite(L, &tm->type, sizeof(tsetlin_type_t), 1, fh);
  tk_lua_fwrite(L, &persist_state, sizeof(bool), 1, fh);
  switch (tm->type) {
    case TM_CLASSIFIER:
      tk_tsetlin_persist_classifier(L, tm->classifier, fh, persist_state);
      break;
    case TM_ENCODER:
      tk_tsetlin_persist_encoder(L, tm->encoder, fh, persist_state);
      break;
    case TM_AUTO_ENCODER:
      tk_tsetlin_persist_auto_encoder(L, tm->auto_encoder, fh, persist_state);
      break;
    case TM_REGRESSOR:
      tk_tsetlin_persist_regressor(L, tm->regressor, fh, persist_state);
      break;
    default:
      return luaL_error(L, "unexpected tsetlin machine type in persist");
  }
  if (!tostr) {
    tk_lua_fclose(L, fh);
    return 0;
  } else {
    size_t len;
    char *data = tk_lua_fslurp(L, fh, &len);
    if (data) {
      lua_pushlstring(L, data, len);
      free(data);
      tk_lua_fclose(L, fh);
      return 1;
    } else {
      tk_lua_fclose(L, fh);
      return 0;
    }
  }
}

static inline void _tk_tsetlin_load_classifier (lua_State *L, tsetlin_classifier_t *tm, FILE *fh, bool read_state, bool has_state)
{
  tk_lua_fread(L, &tm->classes, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->features, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->clauses, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->threshold, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->state_bits, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->boost_true_positive, sizeof(bool), 1, fh);
  tk_lua_fread(L, &tm->input_bits, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->input_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->clause_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->state_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->action_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->filter, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->specificity_low, sizeof(double), 1, fh);
  tk_lua_fread(L, &tm->specificity_high, sizeof(double), 1, fh);
  tm->state = read_state ? malloc(sizeof(unsigned int) * tm->state_chunks) : NULL;
  tm->actions = malloc(sizeof(unsigned int) * tm->action_chunks);
  tm->active_clause = read_state ? malloc(sizeof(unsigned int) * tm->clause_chunks) : NULL;
  tm->specificity = read_state ? malloc(sizeof(double) * tm->clauses) : NULL;
  if (read_state)
    for (unsigned int i = 0; i < tm->clauses; i ++)
      tm->specificity[i] = (1.0 * i / tm->clauses) * (tm->specificity_high - tm->specificity_low) + tm->specificity_low;
  tm->locks = read_state ? malloc(sizeof(pthread_mutex_t) * tm->action_chunks) : NULL;
  if (read_state)
    for (unsigned int i = 0; i < tm->action_chunks; i ++)
      pthread_mutex_init(&tm->locks[i], NULL);
  if (read_state)
    tk_lua_fread(L, tm->state, sizeof(unsigned int), tm->state_chunks, fh);
  else if (has_state)
    tk_lua_fseek(L, sizeof(unsigned int), tm->state_chunks, fh);
  tk_lua_fread(L, tm->actions, sizeof(unsigned int), tm->action_chunks, fh);
}

static inline void tk_tsetlin_load_classifier (lua_State *L, FILE *fh, bool read_state, bool has_state)
{
  tsetlin_t *tm = tk_tsetlin_alloc_classifier(L, read_state);
  _tk_tsetlin_load_classifier(L, tm->classifier, fh, read_state, has_state);
}

static inline void _tk_tsetlin_load_encoder (lua_State *L, tsetlin_encoder_t *en, FILE *fh, bool read_state, bool has_state)
{
  tk_lua_fread(L, &en->encoding_bits, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &en->encoding_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &en->encoding_filter, sizeof(unsigned int), 1, fh);
  _tk_tsetlin_load_classifier(L, &en->encoder, fh, read_state, has_state);
}

static inline void tk_tsetlin_load_encoder (lua_State *L, FILE *fh, bool read_state, bool has_state)
{
  tsetlin_t *tm = tk_tsetlin_alloc_encoder(L, read_state);
  tsetlin_encoder_t *en = tm->encoder;
  _tk_tsetlin_load_encoder(L, en, fh, read_state, has_state);
}

static inline void tk_tsetlin_load_auto_encoder (lua_State *L, FILE *fh, bool read_state, bool has_state)
{
  tsetlin_t *tm = tk_tsetlin_alloc_auto_encoder(L, read_state);
  tsetlin_auto_encoder_t *ae = tm->auto_encoder;
  _tk_tsetlin_load_encoder(L, &ae->encoder, fh, read_state, has_state);
  _tk_tsetlin_load_encoder(L, &ae->decoder, fh, read_state, has_state);
}

static inline void tk_tsetlin_load_regressor (lua_State *L, FILE *fh, bool read_state, bool has_state)
{
  tsetlin_t *tm = tk_tsetlin_alloc_regressor(L, read_state);
  tsetlin_regressor_t *rg = tm->regressor;
  _tk_tsetlin_load_classifier(L, &rg->classifier, fh, read_state, has_state);
}

// TODO: Merge malloc/assignment logic from load_* and create_* to reduce
// changes for coding errors
static inline int tk_tsetlin_load (lua_State *L)
{
  lua_settop(L, 3);
  size_t len;
  const char *data = luaL_checklstring(L, 1, &len);
  bool isstr = lua_type(L, 2) == LUA_TBOOLEAN && lua_toboolean(L, 2);
  FILE *fh = isstr ? tk_lua_fmemopen(L, (char *) data, len, "r") : tk_lua_fopen(L, data, "r");
  bool read_state = lua_toboolean(L, 3);
  tsetlin_type_t type;
  bool has_state;
  tk_lua_fread(L, &type, sizeof(type), 1, fh);
  tk_lua_fread(L, &has_state, sizeof(bool), 1, fh);
  if (read_state && !has_state)
    luaL_error(L, "read_state is true but state not persisted");
  switch (type) {
    case TM_CLASSIFIER:
      tk_tsetlin_load_classifier(L, fh, read_state, has_state);
      tk_lua_fclose(L, fh);
      return 1;
    case TM_ENCODER:
      tk_tsetlin_load_encoder(L, fh, read_state, has_state);
      tk_lua_fclose(L, fh);
      return 1;
    case TM_AUTO_ENCODER:
      tk_tsetlin_load_auto_encoder(L, fh, read_state, has_state);
      tk_lua_fclose(L, fh);
      return 1;
    case TM_REGRESSOR:
      tk_tsetlin_load_regressor(L, fh, read_state, has_state);
      tk_lua_fclose(L, fh);
      return 1;
    default:
      return luaL_error(L, "unexpected tsetlin machine type in load");
  }
}

static inline int tk_tsetlin_type (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  switch (tm->type) {
    case TM_CLASSIFIER:
      lua_pushstring(L, "classifier");
      break;
    case TM_ENCODER:
      lua_pushstring(L, "encoder");
      break;
    case TM_AUTO_ENCODER:
      lua_pushstring(L, "auto_encoder");
      break;
    case TM_REGRESSOR:
      lua_pushstring(L, "regressor");
      break;
    default:
      return luaL_error(L, "unexpected tsetlin machine type in type");
  }
  return 1;
}

static luaL_Reg tk_tsetlin_fns[] =
{

  { "train", tk_tsetlin_train },
  { "evaluate", tk_tsetlin_evaluate },
  { "update", tk_tsetlin_update },
  { "predict", tk_tsetlin_predict },
  { "destroy", tk_tsetlin_destroy },
  { "persist", tk_tsetlin_persist },
  { "type", tk_tsetlin_type },

  { "create", tk_tsetlin_create },
  { "load", tk_tsetlin_load },

  { NULL, NULL }
};

int luaopen_santoku_tsetlin_capi (lua_State *L)
{
  lua_newtable(L); // t
  tk_lua_register(L, tk_tsetlin_fns, 0); // t
  luaL_newmetatable(L, TK_TSETLIN_MT); // t mt
  lua_pushcfunction(L, tk_tsetlin_destroy); // t mt fn
  lua_setfield(L, -2, "__gc"); // t mt
  lua_pop(L, 1); // t
  return 1;
}
