/*

Copyright (C) 2024 Matthew Brooks (Persist to and restore from disk)
Copyright (C) 2024 Matthew Brooks (Lua integration, train/evaluate, active clause)
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

#define _GNU_SOURCE

#include "lua.h"
#include "lauxlib.h"

#include <assert.h>
#include <errno.h>
#include <lauxlib.h>
#include <limits.h>
#include <lua.h>
#include <math.h>
#include <numa.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define TK_TSETLIN_MT "santoku_tsetlin"

typedef __uint128_t tk_bits_t;

#define BITS 128
#define BITS_DIV(x) ((x) >> 7)
#define BITS_MOD(x) ((x) & 127)

const tk_bits_t ALL_MASK = ((tk_bits_t)0xFFFFFFFF << 96) |
                           ((tk_bits_t)0xFFFFFFFF << 64) |
                           ((tk_bits_t)0xFFFFFFFF << 32) |
                           ((tk_bits_t)0xFFFFFFFF);

const tk_bits_t POS_MASK = ((tk_bits_t)0x55555555 << 96) |
                           ((tk_bits_t)0x55555555 << 64) |
                           ((tk_bits_t)0x55555555 << 32) |
                           ((tk_bits_t)0x55555555);

const tk_bits_t NEG_MASK = ((tk_bits_t)0xAAAAAAAA << 96) |
                           ((tk_bits_t)0xAAAAAAAA << 64) |
                           ((tk_bits_t)0xAAAAAAAA << 32) |
                           ((tk_bits_t)0xAAAAAAAA);

typedef enum {
  TM_CLASSIFIER,
  // TM_ENCODER,
} tsetlin_type_t;

typedef enum {
  TM_INIT,
  TM_SETUP,
  TM_DONE,
  TM_PRIME,
  TM_TRAIN,
  TM_PREDICT,
  TM_PREDICT_REDUCE,
} tsetlin_classifier_stage_t;

struct tsetlin_classifier_s;
typedef struct tsetlin_classifier_s tsetlin_classifier_t;

typedef struct {

  tsetlin_classifier_t *tm;

  unsigned int sfirst;
  unsigned int slast;

  unsigned int cfirst;
  unsigned int clast;

  unsigned int index;
  unsigned int sigid;

  struct {
    unsigned int n;
    tk_bits_t *ps;
    unsigned int *ss;
  } train;

  struct {
    unsigned int n;
    tk_bits_t *ps;
  } predict;

  long int *sums_old; // classes x samples
  long int *sums_local; // classes x samples
  unsigned int *shuffle; // samples
  tk_bits_t *active_clause; // clause_chunks*
  tk_bits_t *clause_output; // clause_chunks*
  long int *scores; // classes x samples

} tsetlin_classifier_thread_t;

typedef struct tsetlin_classifier_s {

  bool trained;
  bool destroyed;

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
  tk_bits_t *state; // class clause bit chunk
  tk_bits_t *actions; // class clause chunk

  double active;
  double specificity;

  bool created_threads;

  pthread_mutex_t mutex;
  pthread_cond_t cond_stage;
  pthread_cond_t cond_done;
  unsigned int n_threads;
  unsigned int n_threads_done;
  pthread_t *threads;
  tsetlin_classifier_stage_t stage;
  tsetlin_classifier_thread_t *thread_data;
  size_t results_len;
  unsigned int *results;

  size_t sums_len;
  atomic_long *sums; // samples x classes
  unsigned int sigid;

} tsetlin_classifier_t;

typedef struct {
  tsetlin_type_t type;
  bool has_state;
  union {
    tsetlin_classifier_t *classifier;
    // tsetlin_encoder_t *encoder;
  };
} tsetlin_t;

#define tm_state_old_sum(tm, thread, class, sample) \
  ((tm)->thread_data[thread].sums_old[ \
    (class) * (tm)->thread_data[(thread)].train.n + \
    (sample)])

#define tm_state_sum_local(tm, thread, class, sample) \
  ((tm)->thread_data[thread].sums_local[ \
    (class) * (tm)->thread_data[(thread)].train.n + \
    (sample)])

#define tm_state_clause_output(tm, thread) \
  ((tm)->thread_data[thread].clause_output)

#define tm_state_active_clause(tm, thread) \
  ((tm)->thread_data[thread].active_clause)

#define tm_state_shuffle(tm, thread) \
  ((tm)->thread_data[thread].shuffle)

#define tm_state_scores(tm, thread, class, sample) \
  ((tm)->thread_data[thread].scores[ \
    (class) * (tm)->thread_data[(thread)].predict.n + \
    (sample)])

#define tm_state_clause_chunks(tm, thread) \
  ((tm)->thread_data[thread].clast - (tm)->thread_data[thread].cfirst + 1)

#define tm_state_sum(tm, class, sample) \
  (&(tm)->sums[(sample) * (tm)->classes + \
               (class)])

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

static inline uint64_t mix64 (uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

static inline void seed_rand (unsigned int r) {
  uint64_t raw = (uint64_t)pthread_self() ^ (uint64_t)time(NULL) ^ ((uint64_t) r << 32);
  mcg_state = mix64(raw);
}

static inline double fast_drand ()
{
  return ((double)fast_rand()) / ((double)UINT32_MAX);
}

static inline double fast_index (unsigned int n)
{
  return fast_rand() % n;
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

static inline unsigned int popcount (tk_bits_t x) {
  return
    (unsigned int) __builtin_popcountll((uint64_t)(x >> 64)) +
    (unsigned int) __builtin_popcountll((uint64_t)(x));
}

static inline unsigned int hamming (
  tk_bits_t *a,
  tk_bits_t *b,
  unsigned int bits
) {
  unsigned int chunks = BITS_DIV(bits - 1) + 1;
  unsigned int distance = 0;
  for (unsigned int i = 0; i < chunks; i ++) {
    tk_bits_t diff = a[i] ^ b[i];
    distance += (unsigned int) popcount(diff);
  }
  return distance;
}

static inline double hamming_loss (
  tk_bits_t *a,
  tk_bits_t *b,
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
  tk_bits_t *a,
  tk_bits_t *n,
  tk_bits_t *p,
  unsigned int bits,
  double margin,
  double alpha
) {
  double dist_an = (double) hamming(a, n, bits) / (double) bits;
  double dist_ap = (double) hamming(a, p, bits) / (double) bits;
  return pow(fmin(1.0, fmax(0.0, (double) dist_ap - dist_an + margin)), alpha);
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

static inline int tk_lua_absindex (lua_State *L, int i)
{
  if (i < 0 && i > LUA_REGISTRYINDEX)
    i += lua_gettop(L) + 1;
  return i;
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

static inline void *tk_realloc (
  lua_State *L,
  void *p,
  size_t s
) {
  p = realloc(p, s);
  if (!p) {
    tk_error(L, "realloc failed", ENOMEM);
    return NULL;
  } else {
    return p;
  }
}

static inline void *tk_malloc_interleaved (
  lua_State *L,
  size_t *sp,
  size_t s
) {
  void *p = (numa_available() == -1) ? malloc(s) : numa_alloc_interleaved(s);
  if (!p) {
    tk_error(L, "malloc failed", ENOMEM);
    return NULL;
  } else {
    *sp = s;
    return p;
  }
}

static inline void *tk_ensure_interleaved (
  lua_State *L,
  size_t *s1p,
  void *p0,
  size_t s1,
  bool copy
) {
  size_t s0 = *s1p;
  if (s1 <= s0)
    return p0;
  void *p1 = tk_malloc_interleaved(L, s1p, s1);
  if (!p1) {
    tk_error(L, "realloc failed", ENOMEM);
    return NULL;
  } else {
    if (copy)
      memcpy(p1, p0, s0);
    numa_free(p0, s0);
    return p1;
  }
}

static inline void *tk_malloc_aligned (
  lua_State *L,
  size_t s,
  size_t a
) {
  void *p = NULL;
  if (posix_memalign((void **)&p, a, s) != 0)
    tk_error(L, "malloc failed", ENOMEM);
  return p;
}

static inline void *tk_malloc (
  lua_State *L,
  size_t s
) {
  void *p = malloc(s);
  if (!p) {
    tk_error(L, "malloc failed", ENOMEM);
    return NULL;
  } else {
    return p;
  }
}

static inline int tk_lua_verror (lua_State *L, int n, ...) {
  va_list args;
  va_start(args, n);
  for (int i = 0; i < n; i++) {
    const char *str = va_arg(args, const char *);
    lua_pushstring(L, str);
  }
  va_end(args);
  tk_lua_callmod(L, n, 0, "santoku.error", "error");
  return 0;
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

static inline const char *tk_lua_checkstring (lua_State *L, int i, char *name)
{
  if (lua_type(L, i) != LUA_TSTRING)
    tk_lua_verror(L, 3, name, "value is not a string");
  return luaL_checkstring(L, i);
}

static inline const char *tk_lua_fcheckstring (lua_State *L, int i, char *name, char *field)
{
  lua_getfield(L, i, field);
  if (lua_type(L, -1) != LUA_TSTRING)
    tk_lua_verror(L, 3, name, field, "field is not a string");
  const char *s = luaL_checkstring(L, -1);
  lua_pop(L, 1);
  return s;
}

static inline double tk_lua_checkposdouble (lua_State *L, int i, char *name)
{
  if (lua_type(L, i) != LUA_TNUMBER)
    tk_lua_verror(L, 2, name, "value is not a positive number");
  lua_Number l = luaL_checknumber(L, i);
  if (l < 0)
    tk_lua_verror(L, 2, name, "value is not a positive number");
  return (double) l;
}

static inline double tk_lua_optposdouble (lua_State *L, int i, double def, char *name)
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

static inline double tk_lua_fcheckposdouble (lua_State *L, int i, char *name, char *field)
{
  lua_getfield(L, i, field);
  if (lua_type(L, -1) != LUA_TNUMBER)
    tk_lua_verror(L, 3, name, field, "field is not a positive number");
  lua_Number l = luaL_checknumber(L, -1);
  if (l < 0)
    tk_lua_verror(L, 3, name, field, "field is not a positive number");
  lua_pop(L, 1);
  return l;
}

static inline double tk_lua_foptposdouble (lua_State *L, int i, bool def, char *name, char *field)
{
  lua_getfield(L, i, field);
  if (lua_type(L, -1) == LUA_TNIL) {
    lua_pop(L, 1);
    return def;
  }
  if (lua_type(L, -1) != LUA_TNUMBER)
    tk_lua_verror(L, 3, name, field, "field is not a positive number");
  lua_Number l = luaL_checknumber(L, -1);
  if (l < 0)
    tk_lua_verror(L, 3, name, field, "field is not a positive number");
  lua_pop(L, 1);
  return l;
}

static inline unsigned int tk_lua_checkunsigned (lua_State *L, int i, char *name)
{
  if (lua_type(L, i) != LUA_TNUMBER)
    tk_lua_verror(L, 2, name, "value is not a positive integer");
  lua_Integer l = luaL_checkinteger(L, i);
  if (l < 0)
    tk_lua_verror(L, 2, name, "value is not a positive integer");
  if (l > UINT_MAX)
    luaL_error(L, "value is too large");
  return (unsigned int) l;
}

static inline unsigned int tk_lua_fcheckunsigned (lua_State *L, int i, char *name, char *field)
{
  lua_getfield(L, i, field);
  if (lua_type(L, -1) != LUA_TNUMBER)
    tk_lua_verror(L, 3, name, field, "field is not a positive integer");
  lua_Integer l = luaL_checkinteger(L, -1);
  if (l < 0)
    tk_lua_verror(L, 3, name, field, "field is not a positive integer");
  lua_pop(L, 1);
  return l;
}

static inline bool tk_lua_fcheckboolean (lua_State *L, int i, char *name, char *field)
{
  lua_getfield(L, i, field);
  if (lua_type(L, -1) != LUA_TBOOLEAN)
    tk_lua_verror(L, 3, name, field, "field is not a boolean");
  luaL_checktype(L, -1, LUA_TBOOLEAN);
  bool n = lua_toboolean(L, -1);
  lua_pop(L, 1);
  return n;
}

static inline unsigned int tk_lua_foptunsigned (lua_State *L, int i, bool def, char *name, char *field)
{
  lua_getfield(L, i, field);
  if (lua_type(L, -1) == LUA_TNIL) {
    lua_pop(L, 1);
    return def;
  }
  if (lua_type(L, -1) != LUA_TNUMBER)
    tk_lua_verror(L, 3, name, field, "field is not a positive integer");
  lua_Integer l = luaL_checkinteger(L, -1);
  if (l < 0)
    tk_lua_verror(L, 3, name, field, "field is not a positive integer");
  lua_pop(L, 1);
  return l;
}

static inline unsigned int tk_lua_optunsigned (lua_State *L, int i, unsigned int def, char *name)
{
  if (lua_type(L, i) == LUA_TNIL)
    return def;
  if (lua_type(L, i) != LUA_TNUMBER)
    tk_lua_verror(L, 2, name, "value is not a positive integer");
  lua_Integer l = luaL_checkinteger(L, i);
  if (l < 0)
    tk_lua_verror(L, 2, name, "value is not a positive integer");
  lua_pop(L, 1);
  return l;
}

static inline bool tk_lua_optboolean (lua_State *L, int i, bool def, char *name)
{
  if (lua_type(L, i) == LUA_TNIL)
    return def;
  if (lua_type(L, i) != LUA_TBOOLEAN)
    tk_lua_verror(L, 3, name, "value is not a boolean");
  return lua_toboolean(L, i);
}

static inline bool tk_lua_foptboolean (lua_State *L, int i, bool def, char *name, char *field)
{
  lua_getfield(L, i, field);
  if (lua_type(L, -1) == LUA_TNIL) {
    lua_pop(L, 1);
    return def;
  }
  if (lua_type(L, -1) != LUA_TBOOLEAN)
    tk_lua_verror(L, 3, name, field, "field is not a boolean or nil");
  bool b = lua_toboolean(L, -1);
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

static inline void tk_tsetlin_init_streams (
  tsetlin_classifier_t *tm,
  tk_bits_t *feedback_to_la,
  double specificity
) {
  const unsigned int features = tm->features;
  const unsigned int input_chunks = tm->input_chunks;
  for (unsigned int i = 0; i < input_chunks; i ++)
    feedback_to_la[i] = (tk_bits_t)0;
  const unsigned int n = 2 * features;
  const double p = 1.0 / specificity;
  long int active = normal(n * p, n * p * (1 - p));
  if (active >= n) active = n;
  if (active < 0) active = 0;
  for (unsigned int i = 0; i < active; i++) {
    unsigned int r = fast_rand();
    unsigned int f = ((uint64_t)r * n) >> 32;
    unsigned int chunk = BITS_DIV(f);
    unsigned int bit = BITS_MOD(f);
    // Note: This doesn't seem to add much except extra runtime..
    // while (feedback_to_la[chunk] & ((tk_bits_t)1 << bit)) {
    //   r = fast_rand();
    //   f = ((uint64_t)r * n) >> 32;
    //   chunk = BITS_DIV(f);
    //   bit = BITS_MOD(f);
    // }
    feedback_to_la[chunk] |= ((tk_bits_t)1 << bit);
  }
}

static inline void tm_inc (
  tsetlin_classifier_t *tm,
  unsigned int class,
  unsigned int clause,
  unsigned int chunk,
  tk_bits_t active
) {
  if (!active) return;
  unsigned int m = tm->state_bits - 1;
  tk_bits_t *counts = tm_state_counts(tm, class, clause, chunk);
  tk_bits_t carry, carry_next;
  carry = active;
  for (unsigned int b = 0; b < m; b ++) {
    carry_next = counts[b] & carry;
    counts[b] ^= carry;
    carry = carry_next;
  }
  tk_bits_t *actions = tm_state_actions(tm, class, clause);
  carry_next = actions[chunk] & carry;
  actions[chunk] ^= carry;
  carry = carry_next;
  for (unsigned int b = 0; b < m; b ++)
    counts[b] |= carry;
  actions[chunk] |= carry;
}

static inline void tm_dec (
  tsetlin_classifier_t *tm,
  unsigned int class,
  unsigned int clause,
  unsigned int chunk,
  tk_bits_t active
) {
  if (!active) return;
  unsigned int m = tm->state_bits - 1;
  tk_bits_t *counts = tm_state_counts(tm, class, clause, chunk);
  tk_bits_t carry, carry_next;
  carry = active;
  for (unsigned int b = 0; b < m; b ++) {
    carry_next = (~counts[b]) & carry;
    counts[b] ^= carry;
    carry = carry_next;
  }
  tk_bits_t *actions = tm_state_actions(tm, class, clause);
  carry_next = (~actions[chunk]) & carry;
  actions[chunk] ^= carry;
  carry = carry_next;
  for (unsigned int b = 0; b < m; b ++)
    counts[b] &= ~carry;
  actions[chunk] &= ~carry;
}

static inline void tk_tsetlin_calculate (
  tsetlin_classifier_t *tm,
  unsigned int class,
  tk_bits_t *input,
  bool predict,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  tk_bits_t *clause_output = tm_state_clause_output(tm, thread);
  unsigned int input_chunks = tm->input_chunks;
  for (unsigned int clause_chunk = cfirst; clause_chunk <= clast; clause_chunk ++) {
    clause_output[clause_chunk - cfirst] = 0;
    for (unsigned int clause_chunk_pos = 0; clause_chunk_pos < BITS; clause_chunk_pos ++) {
      tk_bits_t output = 0;
      tk_bits_t all_exclude = 0;
      unsigned int clause = clause_chunk * BITS + clause_chunk_pos;
      tk_bits_t *actions = tm_state_actions(tm, class, clause);
      for (unsigned int k = 0; k < input_chunks - 1; k ++) {
        output |= ((actions[k] & input[k]) ^ actions[k]);
        all_exclude |= actions[k];
      }
      output |=
        (actions[input_chunks - 1] & input[input_chunks - 1]) ^
        (actions[input_chunks - 1]);
      all_exclude |= ((actions[input_chunks - 1]) ^ 0);
      output = !output && !(predict && !all_exclude);
      if (output)
        clause_output[clause_chunk - cfirst] |= ((tk_bits_t)1 << clause_chunk_pos);
    }
  }
}

static inline long int tk_tsetlin_sums (
  tsetlin_classifier_t *tm,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  long int class_sum = 0;
  tk_bits_t *clause_output = tm_state_clause_output(tm, thread);
  for (unsigned int i = cfirst; i <= clast; i ++) {
    class_sum += popcount(clause_output[i - cfirst] & POS_MASK); // 0101
    class_sum -= popcount(clause_output[i - cfirst] & NEG_MASK); // 1010
  }
  return class_sum;
}

static inline void tm_update (
  tsetlin_classifier_t *tm,
  unsigned int class,
  tk_bits_t *input,
  unsigned int s,
  unsigned int target,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  unsigned int clause_chunks = tm_state_clause_chunks(tm, thread);
  long int threshold = (long int) tm->threshold;
  double specificity = tm->specificity;
  unsigned int input_chunks = tm->input_chunks;
  unsigned int boost_true_positive = tm->boost_true_positive;
  long int tgt = (long int) target;
  atomic_long *class_sump = tm_state_sum(tm, class, s);
  tk_bits_t *clause_output = tm_state_clause_output(tm, thread);
  tk_bits_t *active_clause = tm_state_active_clause(tm, thread);
  // NOTE: This is much faster given the stack allocation of feedback arrays.
  // Can we somehow achieve the same with heap?
  tk_bits_t feedback_to_la[input_chunks];
  tk_bits_t feedback_to_clauses[clause_chunks];
  tk_tsetlin_calculate(tm, class, input, false, cfirst, clast, thread);
  long int old_sum = tm_state_old_sum(tm, thread, class, s);
  long int new_sum = tk_tsetlin_sums(tm, cfirst, clast, thread);
  long int diff = new_sum - old_sum;
  long int class_sum;
  if (diff != 0) {
    tm_state_old_sum(tm, thread, class, s) = new_sum;
    class_sum = tm_state_sum_local(tm, thread, class, s) = atomic_fetch_add(class_sump, diff) + diff;
  } else {
    class_sum = tm_state_sum_local(tm, thread, class, s);
  }
  class_sum = (class_sum > threshold) ? threshold : class_sum;
  class_sum = (class_sum < -threshold) ? -threshold : class_sum;
  long int err = (target == 1) ? (threshold - class_sum) : (threshold + class_sum);
  double p = (double) err / (2.0 * threshold);
  for (unsigned int i = 0; i < clause_chunks; i ++)
    feedback_to_clauses[i] = (tk_bits_t)0;
  for (unsigned int i = 0; i < clause_chunks * BITS; i ++) {
    unsigned int clause_chunk = BITS_DIV(i);
    unsigned int clause_chunk_pos = BITS_MOD(i);
    feedback_to_clauses[clause_chunk] |= ((tk_bits_t) fast_chance(p) << clause_chunk_pos);
  }
  for (unsigned int i = 0; i < clause_chunks; i ++)
    feedback_to_clauses[i] &= active_clause[i];
  for (unsigned int j = 0; j < clause_chunks * BITS; j ++) {
    long int jl = (long int) j;
    unsigned int clause_chunk = BITS_DIV(j);
    unsigned int clause_chunk_pos = BITS_MOD(j);
    tk_bits_t *actions = tm_state_actions(tm, class, cfirst * BITS + j);
    bool output = (clause_output[clause_chunk] & ((tk_bits_t)1 << clause_chunk_pos)) > 0;
    if (feedback_to_clauses[clause_chunk] & ((tk_bits_t)1 << clause_chunk_pos)) {
      if ((2 * tgt - 1) * (1 - 2 * (jl & 1)) == -1) {
        // Type II feedback
        if (output)
          for (unsigned int k = 0; k < input_chunks; k ++) {
            tk_bits_t active = (~input[k]) & (~actions[k]);
            tm_inc(tm, class, cfirst * BITS + j, k, active);
          }
      } else if ((2 * tgt - 1) * (1 - 2 * (jl & 1)) == 1) {
        // Type I Feedback
        tk_tsetlin_init_streams(tm, feedback_to_la, specificity);
        if (output) {
          if (boost_true_positive)
            for (unsigned int k = 0; k < input_chunks; k ++) {
              tk_bits_t chunk = input[k];
              tm_inc(tm, class, cfirst * BITS + j, k, chunk);
            }
          else
            for (unsigned int k = 0; k < input_chunks; k ++) {
              tk_bits_t fb = input[k] & (~feedback_to_la[k]);
              tm_inc(tm, class, cfirst * BITS + j, k, fb);
            }
          for (unsigned int k = 0; k < input_chunks; k ++) {
            tk_bits_t fb = (~input[k]) & feedback_to_la[k];
            tm_dec(tm, class, cfirst * BITS + j, k, fb);
          }
        } else {
          for (unsigned int k = 0; k < input_chunks; k ++)
            tm_dec(tm, class, cfirst * BITS + j, k, feedback_to_la[k]);
        }
      }
    }
  }
}

static inline void mc_tm_update (
  tsetlin_classifier_t *tm,
  tk_bits_t *input,
  unsigned int s,
  unsigned int class,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  tm_update(tm, class, input, s, 1, cfirst, clast, thread);
  unsigned int negative_class = (unsigned int) fast_rand() % tm->classes;
  while (negative_class == class)
    negative_class = (unsigned int) fast_rand() % tm->classes;
  tm_update(tm, negative_class, input, s, 0, cfirst, clast, thread);
}

static inline void tk_tsetlin_init_active (
  tsetlin_classifier_t *tm,
  unsigned int thread
) {
  double active = tm->active;
  unsigned int clause_chunks = tm_state_clause_chunks(tm, thread);
  tk_bits_t *active_clause = tm_state_active_clause(tm, thread);
  for (unsigned int i = 0; i < clause_chunks; i ++)
    active_clause[i] = ALL_MASK;
  for (unsigned int i = 0; i < clause_chunks; i ++)
    for (unsigned int j = 0; j < BITS; j ++)
      if (!fast_chance(active))
        active_clause[i] &= ~((tk_bits_t)1 << j);
}

static inline void tk_tsetlin_init_shuffle (
  tsetlin_classifier_t *tm,
  unsigned int thread,
  unsigned int n
) {
  unsigned int *shuffle = tm_state_shuffle(tm, thread);
  for (unsigned int i = 0; i < n; i ++) {
    shuffle[i] = i;
    unsigned int j = i == 0 ? 0 : fast_rand() % (i + 1);
    unsigned int t = shuffle[i];
    shuffle[i] = shuffle[j];
    shuffle[j] = t;
  }
}

tsetlin_t *tk_tsetlin_peek (lua_State *L, int i)
{
  return (tsetlin_t *) luaL_checkudata(L, i, TK_TSETLIN_MT);
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
  tm->classifier = tk_malloc(L, sizeof(tsetlin_classifier_t));
  memset(tm->classifier, 0, sizeof(tsetlin_classifier_t));
  return tm;
}

static inline void tk_tsetlin_wait_for_threads (
  pthread_mutex_t *mutex,
  pthread_cond_t *cond_done,
  unsigned int *n_threads_done,
  unsigned int n_threads
) {
  pthread_mutex_lock(mutex);
  while ((*n_threads_done) < n_threads)
    pthread_cond_wait(cond_done, mutex);
  pthread_mutex_unlock(mutex);
}

static inline void tk_tsetlin_signal (
  int stage,
  unsigned int *sigid,
  int *stagep,
  pthread_mutex_t *mutex,
  pthread_cond_t *cond_stage,
  pthread_cond_t *cond_done,
  unsigned int *n_threads_done,
  unsigned int n_threads
) {
  pthread_mutex_lock(mutex);
  (*sigid) ++;
  (*stagep) = stage;
  (*n_threads_done) = 0;
  pthread_cond_broadcast(cond_stage);
  pthread_mutex_unlock(mutex);
  tk_tsetlin_wait_for_threads(mutex, cond_done, n_threads_done, n_threads);
  pthread_cond_broadcast(cond_stage);
}

static void tk_classifier_setup_thread (tsetlin_classifier_t *, unsigned int, tk_bits_t *, unsigned int, unsigned int, unsigned int);
static void tk_classifier_prime_thread (tsetlin_classifier_t *, unsigned int, tk_bits_t *, unsigned int, unsigned int, unsigned int);
static void tk_classifier_train_thread (tsetlin_classifier_t *, unsigned int, tk_bits_t *, unsigned int *, unsigned int, unsigned int, unsigned int);
static void tk_classifier_predict_thread (tsetlin_classifier_t *, unsigned int, tk_bits_t *, unsigned int, unsigned int, unsigned int);
static void tk_classifier_predict_reduce_thread (tsetlin_classifier_t *, unsigned int, unsigned int, unsigned int);

static void *tk_tsetlin_classifier_worker (void *datap)
{
  tsetlin_classifier_thread_t *data =
    (tsetlin_classifier_thread_t *) datap;
  pthread_mutex_lock(&data->tm->mutex);
  data->tm->n_threads_done ++;
  if (data->tm->n_threads_done == data->tm->n_threads)
    pthread_cond_signal(&data->tm->cond_done);
  pthread_mutex_unlock(&data->tm->mutex);
  while (1) {
    pthread_mutex_lock(&data->tm->mutex);
    while (data->sigid == data->tm->sigid)
      pthread_cond_wait(&data->tm->cond_stage, &data->tm->mutex);
    data->sigid = data->tm->sigid;
    tsetlin_classifier_stage_t stage = data->tm->stage;
    pthread_mutex_unlock(&data->tm->mutex);
    switch (stage) {
      case TM_SETUP:
        tk_classifier_setup_thread(
          data->tm,
          data->train.n,
          data->train.ps,
          data->cfirst,
          data->clast,
          data->index);
        break;
      case TM_PRIME:
        tk_classifier_prime_thread(
          data->tm,
          data->train.n,
          data->train.ps,
          data->cfirst,
          data->clast,
          data->index);
        break;
      case TM_TRAIN:
        tk_classifier_train_thread(
          data->tm,
          data->train.n,
          data->train.ps,
          data->train.ss,
          data->cfirst,
          data->clast,
          data->index);
        break;
      case TM_PREDICT:
        tk_classifier_predict_thread(
          data->tm,
          data->predict.n,
          data->predict.ps,
          data->cfirst,
          data->clast,
          data->index);
        break;
      case TM_PREDICT_REDUCE:
        tk_classifier_predict_reduce_thread(
          data->tm,
          data->predict.n,
          data->sfirst,
          data->slast);
        break;
      case TM_DONE:
        break;
      default:
        assert(false);
        break;
    }
    pthread_mutex_lock(&data->tm->mutex);
    data->tm->n_threads_done ++;
    if (data->tm->n_threads_done == data->tm->n_threads)
      pthread_cond_signal(&data->tm->cond_done);
    pthread_mutex_unlock(&data->tm->mutex);
    if (stage == TM_DONE)
      break;
  }
  return NULL;
}

static inline void tk_pin_thread_to_cpu (
  unsigned int thread_index,
  unsigned int n_threads
) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  unsigned int n_nodes = (unsigned int) numa_max_node() + 1;
  unsigned int threads_per_node = n_threads / n_nodes;
  if (threads_per_node == 0) threads_per_node = 1;
  unsigned int node = thread_index / threads_per_node;
  if (node >= n_nodes) node = n_nodes - 1;
  struct bitmask *cpus = numa_allocate_cpumask();
  if (numa_node_to_cpus((int) node, cpus) == 0) {
    unsigned int count = 0;
    for (unsigned int i = 0; i < cpus->size; ++i) {
      if (numa_bitmask_isbitset(cpus, i)) {
        count ++;
      }
    }
    if (count > 0) {
      unsigned int local_index =
        (thread_index - node * threads_per_node) % count;
      unsigned int found = 0;
      for (unsigned int i = 0; i < cpus->size; ++i) {
        if (numa_bitmask_isbitset(cpus, i)) {
          if (found == local_index) {
            CPU_SET(i, &cpuset);
            break;
          }
          found ++;
        }
      }
    }
  }
  numa_free_cpumask(cpus);
  pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
}

static void *tk_tsetlin_classifier_worker_wrapper (void *arg) {
  tsetlin_classifier_thread_t *td = (tsetlin_classifier_thread_t *)arg;
  if (numa_available() != -1 && numa_max_node() > 0)
    tk_pin_thread_to_cpu(td->index, td->tm->n_threads);
  return tk_tsetlin_classifier_worker(arg);
}

static inline void tk_tsetlin_setup_threads (
  lua_State *L,
  tsetlin_classifier_t *tm,
  unsigned int n_threads
) {
  tm->n_threads = n_threads;
  tm->n_threads_done = 0;
  tm->stage = TM_INIT;
  tm->threads = tk_malloc(L, tm->n_threads * sizeof(pthread_t));
  tm->thread_data = tk_malloc(L, tm->n_threads * sizeof(tsetlin_classifier_thread_t));
  memset(tm->thread_data, 0, tm->n_threads * sizeof(tsetlin_classifier_thread_t));

  // TODO: check errors
  pthread_mutex_init(&tm->mutex, NULL);
  pthread_cond_init(&tm->cond_stage, NULL);
  pthread_cond_init(&tm->cond_done, NULL);

  for (unsigned int i = 0; i < tm->n_threads; i ++) {
    tm->thread_data[i].tm = tm;
    tm->thread_data[i].sigid = 0;
    tm->thread_data[i].index = i;
    if (!tm->created_threads && pthread_create(&tm->threads[i], NULL, tk_tsetlin_classifier_worker_wrapper, &tm->thread_data[i]) != 0)
      tk_error(L, "pthread_create", errno);
  }

  unsigned int cslice = tm->clause_chunks / tm->n_threads;
  unsigned int cremaining = tm->clause_chunks % tm->n_threads;
  unsigned int cfirst = 0;
  for (unsigned int i = 0; i < tm->n_threads; i ++) {
    unsigned int extra = cremaining ? 1 : 0;
    unsigned int count = cslice + extra;
    if (cremaining) cremaining--;
    tm->thread_data[i].cfirst = cfirst;
    tm->thread_data[i].clast = cfirst + count - 1;
    cfirst += count;
  }

  for (unsigned int i = 0; i < tm->n_threads; i ++) {
    unsigned int clause_chunks = tm_state_clause_chunks(tm, i);
    tm->thread_data[i].active_clause = tk_malloc_aligned(L, sizeof(tk_bits_t) * clause_chunks, BITS);
    tm->thread_data[i].clause_output = tk_malloc_aligned(L, sizeof(tk_bits_t) * clause_chunks, BITS);
  }

  tm->created_threads = true;
  tk_tsetlin_wait_for_threads(&tm->mutex, &tm->cond_done, &tm->n_threads_done, tm->n_threads);
}

static inline void tk_tsetlin_init_classifier (
  lua_State *L,
  tsetlin_classifier_t *tm,
  unsigned int classes,
  unsigned int features,
  unsigned int clauses,
  unsigned int state_bits,
  double thresholdf,
  bool boost_true_positive,
  double specificity,
  unsigned int n_threads
) {
  if (!classes)
    tk_lua_verror(L, 3, "create classifier", "classes", "must be greater than 1");
  if (!clauses)
    tk_lua_verror(L, 3, "create classifier", "clauses", "must be greater than 0");
  if (BITS_MOD(clauses))
    tk_lua_verror(L, 3, "create classifier", "clauses", "must be a multiple of 128");
  if (BITS_MOD(features))
    tk_lua_verror(L, 3, "create classifier", "features", "must be a multiple of 128");
  if (state_bits < 2)
    tk_lua_verror(L, 3, "create classifier", "bits", "must be greater than 1");
  if (thresholdf <= 0)
    tk_lua_verror(L, 3, "create classifier", "target", "must be greater than 0");
  tm->classes = classes;
  tm->clauses = clauses;
  tm->threshold = ceil(thresholdf >= 1 ? thresholdf : fmaxf(1.0, (double) clauses / thresholdf));
  tm->features = features;
  tm->state_bits = state_bits;
  tm->boost_true_positive = boost_true_positive;
  tm->input_bits = 2 * tm->features;
  tm->input_chunks = BITS_DIV((tm->input_bits - 1)) + 1;
  tm->clause_chunks = BITS_DIV((tm->clauses - 1)) + 1;
  tm->state_chunks = tm->classes * tm->clauses * (tm->state_bits - 1) * tm->input_chunks;
  tm->action_chunks = tm->classes * tm->clauses * tm->input_chunks;
  tm->state = tk_malloc_aligned(L, sizeof(tk_bits_t) * tm->state_chunks, BITS);
  tm->actions = tk_malloc_aligned(L, sizeof(tk_bits_t) * tm->action_chunks, BITS);
  tm->sigid = 0;
  tm->specificity = specificity;
  if (!(tm->state && tm->actions))
    luaL_error(L, "error in malloc during creation of classifier");
  tk_tsetlin_setup_threads(L, tm, n_threads);
}

static inline unsigned int tk_tsetlin_get_nthreads (
  lua_State *L, int i, char *name, char *field
) {
  long ts;
  unsigned int n_threads;
  if (field != NULL)
    n_threads = tk_lua_foptunsigned(L, i, 0, name, field);
  else
    n_threads = tk_lua_optunsigned(L, i, 0, name);
  if (n_threads)
    return n_threads;
  ts = sysconf(_SC_NPROCESSORS_ONLN) - 1;
  if (ts <= 0)
    return (unsigned int) tk_lua_verror(L, 3, name, "sysconf", errno);
  lua_pushinteger(L, ts);
  n_threads = tk_lua_checkunsigned(L, -1, "sysconf");
  lua_pop(L, 1);
  return n_threads;
}

static inline void tk_tsetlin_create_classifier (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_alloc_classifier(L, true);
  lua_insert(L, 1);

  tk_tsetlin_init_classifier(L, tm->classifier,
      tk_lua_fcheckunsigned(L, 2, "create classifier", "classes"),
      tk_lua_fcheckunsigned(L, 2, "create classifier", "features"),
      tk_lua_fcheckunsigned(L, 2, "create classifier", "clauses"),
      tk_lua_fcheckunsigned(L, 2, "create classifier", "state"),
      tk_lua_fcheckposdouble(L, 2, "create classifier", "target"),
      tk_lua_fcheckboolean(L, 2, "create classifier", "boost"),
      tk_lua_fcheckposdouble(L, 2, "create classifier", "specificity"),
      tk_tsetlin_get_nthreads(L, 2, "create classifier", "threads"));

  lua_settop(L, 1);
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

  // } else if (!strcmp(type, "encoder")) {

  //   lua_remove(L, 1);
  //   tk_tsetlin_create_encoder(L);
  //   return 1;

  } else {

    luaL_error(L, "unexpected tsetlin machine type in create");
    return 0;

  }
}

// TODO: expose via the api
static inline void tk_classifier_shrink (tsetlin_classifier_t *tm)
{
  if (tm == NULL) return;
  free(tm->state); tm->state = NULL;
  if (numa_available() == -1) {
    free(tm->sums); tm->sums = NULL;
  } else {
    numa_free(tm->sums, tm->sums_len); tm->sums = NULL; tm->sums_len = 0;
  }
  if (tm->created_threads)
    for (unsigned int i = 0; i < tm->n_threads; i ++) {
      free(tm->thread_data[i].sums_old); tm->thread_data[i].sums_old = NULL;
      free(tm->thread_data[i].sums_local); tm->thread_data[i].sums_local = NULL;
      free(tm->thread_data[i].shuffle); tm->thread_data[i].shuffle = NULL;
    }
}

static inline void tk_classifier_destroy (tsetlin_classifier_t *tm)
{
  if (tm == NULL) return;
  if (tm->destroyed) return;
  tm->destroyed = true;
  tk_classifier_shrink(tm);
  free(tm->actions); tm->actions = NULL;
  if (numa_available() == -1) {
    free(tm->results); tm->results = NULL;
  } else {
    numa_free(tm->results, tm->results_len); tm->results = NULL; tm->results_len = 0;
  }
  tk_tsetlin_signal(
    (int) TM_DONE, &tm->sigid,
    (int *) &tm->stage, &tm->mutex, &tm->cond_stage, &tm->cond_done,
    &tm->n_threads_done, tm->n_threads);
  // TODO: What is the right way to deal with potential thread errors (or other
  // errors, for that matter) during the finalizer?
  if (tm->created_threads)
    for (unsigned int i = 0; i < tm->n_threads; i ++) {
      free(tm->thread_data[i].scores); tm->thread_data[i].scores = NULL;
      free(tm->thread_data[i].clause_output); tm->thread_data[i].clause_output = NULL;
      pthread_join(tm->threads[i], NULL);
      // if (pthread_join(tm->threads[i], NULL) != 0)
      //   tk_error(L, "pthread_join", errno);
    }
  pthread_mutex_destroy(&tm->mutex);
  pthread_cond_destroy(&tm->cond_stage);
  pthread_cond_destroy(&tm->cond_done);
  free(tm->threads); tm->threads = NULL;
  free(tm->thread_data); tm->thread_data = NULL;
}

static inline int tk_tsetlin_destroy (lua_State *L)
{
  lua_settop(L, 1);
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  switch (tm->type) {
    case TM_CLASSIFIER:
      if (tm->classifier) {
        tk_classifier_destroy(tm->classifier);
        free(tm->classifier);
        tm->classifier = NULL;
      }
      break;
    // case TM_ENCODER:
    //   if (tm->encoder) {
    //     tk_tsetlin_destroy_classifier(&tm->encoder->encoder);
    //     free(tm->encoder);
    //     tm->encoder = NULL;
    //   }
    //   break;
    default:
      return luaL_error(L, "unexpected tsetlin machine type in destroy");
  }
  return 0;
}

static void tk_classifier_predict_reduce_thread (
  tsetlin_classifier_t *tm,
  unsigned int n,
  unsigned int sfirst,
  unsigned int slast
) {
  unsigned int *results = tm->results;
  for (unsigned int s = sfirst; s <= slast; s ++) {
    long int maxval = -INT64_MAX;
    unsigned int maxclass = 0;
    for (unsigned int class = 0; class < tm->classes; class ++) {
      long int sum = 0;
      for (unsigned int t = 0; t < tm->n_threads; t ++)
        sum += tm_state_scores(tm, t, class, s);
      if (sum > maxval) {
        maxval = sum;
        maxclass = class;
      }
    }
    results[s] = maxclass;
  }
}

static void tk_classifier_predict_thread (
  tsetlin_classifier_t *tm,
  unsigned int n,
  tk_bits_t *ps,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  unsigned int input_chunks = tm->input_chunks;
  for (unsigned int class = 0; class < tm->classes; class ++) {
    for (unsigned int s = 0; s < n; s ++) {
      tk_tsetlin_calculate(tm, class, ps + s * input_chunks, true, cfirst, clast, thread);
      tm_state_scores(tm, thread, class, s) = tk_tsetlin_sums(tm, cfirst, clast, thread);
    }
  }
}

static inline void tk_tsetlin_setup_thread_samples (
  tsetlin_classifier_t *tm,
  unsigned int n
) {
  unsigned int sslice = n / tm->n_threads;
  unsigned int sremaining = n % tm->n_threads;
  unsigned int sfirst = 0;
  for (unsigned int i = 0; i < tm->n_threads; i ++) {
    unsigned int extra = sremaining ? 1 : 0;
    unsigned int count = sslice + extra;
    if (sremaining) sremaining--;
    tm->thread_data[i].sfirst = sfirst;
    tm->thread_data[i].slast = sfirst + count - 1;
    sfirst += count;
  }
}

static inline int tk_tsetlin_predict_classifier (
  lua_State *L,
  tsetlin_classifier_t *tm
) {

  lua_settop(L, 2);
  tk_bits_t *ps = (tk_bits_t *) tk_lua_checkstring(L, 2, "argument 1 is not a raw bit-matrix of samples");
  unsigned int n = tk_lua_checkunsigned(L, 3, "argument 2 is not an integer n_samples");

  tm->results = tk_ensure_interleaved(L, &tm->results_len, tm->results, n * sizeof(unsigned int), false);

  for (unsigned int i = 0; i < tm->n_threads; i ++) {
    tm->thread_data[i].predict.n = n;
    tm->thread_data[i].predict.ps = ps;
    tm->thread_data[i].scores = tk_realloc(L, tm->thread_data[i].scores, tm->classes * n * sizeof(long int));
  }

  tk_tsetlin_setup_thread_samples(tm, n);

  tk_tsetlin_signal(
    (int) TM_PREDICT, &tm->sigid,
    (int *) &tm->stage, &tm->mutex, &tm->cond_stage, &tm->cond_done,
    &tm->n_threads_done, tm->n_threads);

  tk_tsetlin_signal(
    (int) TM_PREDICT_REDUCE, &tm->sigid,
    (int *) &tm->stage, &tm->mutex, &tm->cond_stage, &tm->cond_done,
    &tm->n_threads_done, tm->n_threads);

  lua_pushlstring(L, (char *) tm->results, n * sizeof(unsigned int));
  return 1;
}

static inline int tk_tsetlin_predict (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  switch (tm->type) {
    case TM_CLASSIFIER:
      return tk_tsetlin_predict_classifier(L, tm->classifier);
    // case TM_ENCODER:
    //   return tk_tsetlin_predict_encoder(L, tm->encoder);
    default:
      return luaL_error(L, "unexpected tsetlin machine type in predict");
  }
  return 0;
}

static void tk_classifier_setup_thread (
  tsetlin_classifier_t *tm,
  unsigned int n,
  tk_bits_t *ps,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  for (unsigned int class = 0; class < tm->classes; class ++)
    for (unsigned int clause_chunk = cfirst; clause_chunk <= clast; clause_chunk ++)
      for (unsigned int clause_chunk_pos = 0; clause_chunk_pos < BITS; clause_chunk_pos ++) {
        unsigned int clause = clause_chunk * BITS + clause_chunk_pos;
        for (unsigned int input_chunk = 0; input_chunk < tm->input_chunks; input_chunk ++) {
          unsigned int m = tm->state_bits - 1;
          tk_bits_t *actions = tm_state_actions(tm, class, clause);
          tk_bits_t *counts = tm_state_counts(tm, class, clause, input_chunk);
          actions[input_chunk] = 0;
          for (unsigned int b = 0; b < m; b ++)
            counts[b] = (tk_bits_t)~(tk_bits_t)0;
        }
      }
  for (unsigned int s = 0; s < n; s ++) {
    for (unsigned int class = 0; class < tm->classes; class ++) {
      atomic_long *class_sump = tm_state_sum(tm, class, s);
      tk_tsetlin_calculate(tm, class, ps + s * tm->input_chunks, false, cfirst, clast, thread);
      long int sum = tk_tsetlin_sums(tm, cfirst, clast, thread);
      atomic_fetch_add(class_sump, sum);
      tm_state_old_sum(tm, thread, class, s) = sum;
    }
  }
}

static void tk_classifier_prime_thread (
  tsetlin_classifier_t *tm,
  unsigned int n,
  tk_bits_t *ps,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  for (unsigned int s = 0; s < n; s ++) {
    for (unsigned int class = 0; class < tm->classes; class ++) {
      atomic_long *class_sump = tm_state_sum(tm, class, s);
      tm_state_sum_local(tm, thread, class, s) = atomic_load(class_sump);
    }
  }
}

static void tk_classifier_train_thread (
  tsetlin_classifier_t *tm,
  unsigned int n,
  tk_bits_t *ps,
  unsigned int *ss,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  seed_rand(cfirst);
  tk_tsetlin_init_active(tm, thread);
  tk_tsetlin_init_shuffle(tm, thread, n);
  unsigned int *shuffle = tm_state_shuffle(tm, thread);
  for (unsigned int i = 0; i < n; i ++) {
    unsigned int s = shuffle[i];
    mc_tm_update(tm, ps + s * tm->input_chunks, s, ss[s], cfirst, clast, thread);
  }
}

static inline int tk_tsetlin_train_classifier (
  lua_State *L,
  tsetlin_classifier_t *tm
) {

  unsigned int n = tk_lua_fcheckunsigned(L, 2, "train", "samples");
  tk_bits_t *ps = (tk_bits_t *) tk_lua_fcheckstring(L, 2, "train", "problems");
  unsigned int *ss = (unsigned int *) tk_lua_fcheckstring(L, 2, "train", "solutions");
  unsigned int max_iter =  tk_lua_fcheckunsigned(L, 2, "train", "iterations");
  tm->active = tk_lua_fcheckposdouble(L, 2, "train", "active");

  int i_each = -1;
  if (tk_lua_ftype(L, 2, "each") != LUA_TNIL) {
    lua_getfield(L, 2, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  for (unsigned int i = 0; i < tm->n_threads; i ++) {
    tm->thread_data[i].train.n = n;
    tm->thread_data[i].train.ps = ps;
    tm->thread_data[i].train.ss = ss;
    tm->thread_data[i].shuffle = tk_realloc(L, tm->thread_data[i].shuffle, n * sizeof(unsigned int));
    tm->thread_data[i].sums_old = tk_realloc(L, tm->thread_data[i].sums_old, tm->classes * n * sizeof(long int));
    tm->thread_data[i].sums_local = tk_realloc(L, tm->thread_data[i].sums_local, tm->classes * n * sizeof(long int));
  }

  tm->sums = tk_ensure_interleaved(L, &tm->sums_len, tm->sums, n * tm->classes * sizeof(atomic_long), false);
  for (unsigned int c = 0; c < tm->classes; c ++)
    for (unsigned int s = 0; s < n; s ++)
      atomic_init(tm_state_sum(tm, c, s), 0);

  tk_tsetlin_setup_thread_samples(tm, n);

  tk_tsetlin_signal(
    (int) TM_SETUP, &tm->sigid,
    (int *) &tm->stage, &tm->mutex, &tm->cond_stage, &tm->cond_done,
    &tm->n_threads_done, tm->n_threads);

  tk_tsetlin_signal(
    (int) TM_PRIME, &tm->sigid,
    (int *) &tm->stage, &tm->mutex, &tm->cond_stage, &tm->cond_done,
    &tm->n_threads_done, tm->n_threads);

  for (unsigned int i = 0; i < max_iter; i ++) {

    // for (unsigned int s = 0; s < 200; s ++)
    //   for (unsigned int c = 0; c < 10; c ++) {
    //     long int x = atomic_load(tm_state_sum(tm, s, c));
    //     if (x < 0)
    //       fprintf(stderr, "> %u %u %ld\n", s, c, x);
    //   }

    tk_tsetlin_signal(
      (int) TM_TRAIN, &tm->sigid,
      (int *) &tm->stage, &tm->mutex, &tm->cond_stage, &tm->cond_done,
      &tm->n_threads_done, tm->n_threads);

    if (i_each > -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, i + 1);
      lua_call(L, 1, 1);
      if (lua_type(L, -1) == LUA_TBOOLEAN && lua_toboolean(L, -1) == 0)
        break;
      lua_pop(L, 1);
    }

  }

  tk_classifier_shrink(tm);

  tm->trained = true;
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
    // case TM_ENCODER:
    //   return tk_tsetlin_train_encoder(L, tm->encoder);
    default:
      return luaL_error(L, "unexpected tsetlin machine type in train");
  }
}

static inline int tk_tsetlin_evaluate_classifier (
  lua_State *L,
  tsetlin_classifier_t *tm
) {

  lua_settop(L, 2);
  unsigned int n = tk_lua_fcheckunsigned(L, 2, "evaluate", "samples");
  tk_bits_t *ps = (tk_bits_t *) tk_lua_fcheckstring(L, 2, "evaluate", "problems");
  unsigned int *ss = (unsigned int *) tk_lua_fcheckstring(L, 2, "evaluate", "solutions");
  bool track_stats = tk_lua_foptboolean(L, 2, false, "evaluate", "stats");

  tm->results = tk_ensure_interleaved(L, &tm->results_len, tm->results, n * sizeof(unsigned int), false);

  for (unsigned int i = 0; i < tm->n_threads; i ++) {
    tm->thread_data[i].predict.n = n;
    tm->thread_data[i].predict.ps = ps;
    tm->thread_data[i].scores = tk_realloc(L, tm->thread_data[i].scores, sizeof(long int) * tm->classes * n);
  }

  tk_tsetlin_setup_thread_samples(tm, n);

  tk_tsetlin_signal(
    (int) TM_PREDICT, &tm->sigid,
    (int *) &tm->stage, &tm->mutex, &tm->cond_stage, &tm->cond_done,
    &tm->n_threads_done, tm->n_threads);

  tk_tsetlin_signal(
    (int) TM_PREDICT_REDUCE, &tm->sigid,
    (int *) &tm->stage, &tm->mutex, &tm->cond_stage, &tm->cond_done,
    &tm->n_threads_done, tm->n_threads);

  unsigned int correct = 0;
  int i_observations = 0;
  int i_predictions = 0;
  int i_confusion = 0;

  if (track_stats) {
    lua_newtable(L);
    lua_newtable(L);
    lua_newtable(L);
    i_observations = tk_lua_absindex(L, -3);
    i_predictions = tk_lua_absindex(L, -2);
    i_confusion = tk_lua_absindex(L, -1);
  }

  // TODO: The count of expected classes can/should be cached. Will require
  // some thinking since evaluate can be called separately from any training
  // loop. Might not be totally necessary to do.

  for (unsigned int s = 0; s < n; s ++) {

    unsigned int expected = ss[s];
    unsigned int predicted = tm->results[s];

    if (expected == predicted)
      correct ++;

    if (!track_stats)
      continue;

    lua_Integer v;

    lua_pushinteger(L, expected); // e
    lua_gettable(L, i_observations); // v
    v = luaL_optinteger(L, -1, 0); // v
    lua_pop(L, 1);
    lua_pushinteger(L, expected); // e
    lua_pushinteger(L, v + 1); // e v
    lua_settable(L, i_observations); //

    lua_pushinteger(L, predicted); // e
    lua_gettable(L, i_predictions); // v
    v = luaL_optinteger(L, -1, 0); // v
    lua_pop(L, 1);
    lua_pushinteger(L, predicted); // e
    lua_pushinteger(L, v + 1); // e v
    lua_settable(L, i_predictions); //

    if (expected != predicted) {
      lua_pushinteger(L, expected); // e
      lua_gettable(L, i_confusion); // t
      if (lua_type(L, -1) == LUA_TNIL) {
        lua_pop(L, 1);
        lua_pushinteger(L, expected); // e
        lua_newtable(L); // e t
        lua_pushvalue(L, -1); // e t t
        lua_insert(L, -3); // t e t
        lua_settable(L, i_confusion); // t
      }
      lua_pushinteger(L, predicted); // t c
      lua_gettable(L, -2); // t v
      v = luaL_optinteger(L, -1, 0); // t c
      lua_pop(L, 1); // t
      lua_pushinteger(L, predicted); // t p
      lua_pushinteger(L, v + 1); // t p c
      lua_settable(L, -3); // t
      lua_pop(L, 1);
    }

  }

  lua_pushnumber(L, correct);

  if (!(track_stats && i_observations && i_predictions && i_confusion))
    return 1;

  lua_pushvalue(L, i_confusion);
  lua_pushvalue(L, i_predictions);
  lua_pushvalue(L, i_observations);
  return 4;
}

static inline int tk_tsetlin_evaluate (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  switch (tm->type) {
    case TM_CLASSIFIER:
      return tk_tsetlin_evaluate_classifier(L, tm->classifier);
    // case TM_ENCODER:
    //   return tk_tsetlin_evaluate_encoder(L, tm->encoder);
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
  tk_lua_fwrite(L, &tm->specificity, sizeof(double), 1, fh);
  tk_lua_fwrite(L, tm->actions, sizeof(tk_bits_t), tm->action_chunks, fh);
}

static inline void tk_tsetlin_persist_classifier (lua_State *L, tsetlin_classifier_t *tm, FILE *fh, bool persist_state)
{
  _tk_tsetlin_persist_classifier(L, tm, fh, persist_state);
}

static inline int tk_tsetlin_persist (lua_State *L)
{
  lua_settop(L, 3);
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  bool tostr = lua_type(L, 2) == LUA_TNIL;
  FILE *fh = tostr ? tk_lua_tmpfile(L) : tk_lua_fopen(L, tk_lua_checkstring(L, 2, "persist path"), "w");
  bool persist_state = lua_toboolean(L, 3);
  if (persist_state && !tm->has_state)
    luaL_error(L, "can't persist the state of a model loaded without state");
  tk_lua_fwrite(L, &tm->type, sizeof(tsetlin_type_t), 1, fh);
  tk_lua_fwrite(L, &persist_state, sizeof(bool), 1, fh);
  switch (tm->type) {
    case TM_CLASSIFIER:
      tk_tsetlin_persist_classifier(L, tm->classifier, fh, persist_state);
      break;
    // case TM_ENCODER:
    //   tk_tsetlin_persist_encoder(L, tm->encoder, fh, persist_state);
    //   break;
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

static inline void _tk_tsetlin_load_classifier (lua_State *L, tsetlin_classifier_t *tm, FILE *fh, bool read_state, bool has_state, unsigned int n_threads)
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
  tk_lua_fread(L, &tm->specificity, sizeof(double), 1, fh);
  tm->actions = tk_malloc_aligned(L, sizeof(tk_bits_t) * tm->action_chunks, BITS);
  tk_lua_fread(L, tm->actions, sizeof(tk_bits_t), tm->action_chunks, fh);
  tk_tsetlin_setup_threads(L, tm, n_threads);
}

static inline void tk_tsetlin_load_classifier (lua_State *L, FILE *fh, bool read_state, bool has_state)
{
  tsetlin_t *tm = tk_tsetlin_alloc_classifier(L, read_state);
  unsigned int n_threads = tk_tsetlin_get_nthreads(L, 2, "threads", NULL);
  _tk_tsetlin_load_classifier(L, tm->classifier, fh, read_state, has_state, n_threads);
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
    // case TM_ENCODER:
    //   tk_tsetlin_load_encoder(L, fh, read_state, has_state);
    //   tk_lua_fclose(L, fh);
    //   return 1;
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
    // case TM_ENCODER:
    //   lua_pushstring(L, "encoder");
    //   break;
    default:
      return luaL_error(L, "unexpected tsetlin machine type in type");
  }
  return 1;
}

static luaL_Reg tk_tsetlin_fns[] =
{

  { "train", tk_tsetlin_train },
  { "evaluate", tk_tsetlin_evaluate },
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

// typedef struct {

//   unsigned int encoding_bits;
//   unsigned int encoding_chunks;
//   unsigned int encoding_filter;

//   tsetlin_classifier_t encoder;

// } tsetlin_encoder_t;

// static inline void en_tm_encode (
//   tsetlin_encoder_t *tm,
//   unsigned int *input,
//   unsigned int *encoding,
//   long int *scores
// ) {
//   unsigned int encoder_classes = tm->encoder.classes;
//   unsigned int clause_output[tm->encoder.clause_chunks];
//   tm_score(&tm->encoder, input, clause_output, scores);
//   for (unsigned int i = 0; i < encoder_classes; i ++)
//   {
//     unsigned int chunk = i / BITS;
//     unsigned int pos = i % BITS;
//     if (scores[i] > 0)
//       encoding[chunk] |= (1U << pos);
//     else
//       encoding[chunk] &= ~(1U << pos);
//   }
//   encoding[tm->encoding_chunks - 1] &= tm->encoder.filter;
// }

// static inline void en_tm_update (
//   tsetlin_encoder_t *tm,
//   unsigned int *a,
//   unsigned int *n,
//   unsigned int *p,
//   unsigned int *clause_output,
//   unsigned int *feedback_to_clauses,
//   unsigned int *feedback_to_la,
//   long int *scores,
//   double margin,
//   double loss_alpha
// ) {

//   tsetlin_classifier_t *encoder = &tm->encoder;
//   unsigned int classes = encoder->classes;
//   unsigned int encoding_chunks = tm->encoding_chunks;
//   unsigned int encoding_bits = tm->encoding_bits;

//   unsigned int encoding_a[encoding_chunks];
//   unsigned int encoding_n[encoding_chunks];
//   unsigned int encoding_p[encoding_chunks];

//   en_tm_encode(tm, a, encoding_a, scores);
//   en_tm_encode(tm, n, encoding_n, scores);
//   en_tm_encode(tm, p, encoding_p, scores);

//   double loss = triplet_loss_hamming(encoding_a, encoding_n, encoding_p, encoding_bits, margin, loss_alpha);

//   if (fast_chance(1 - loss))
//     return;

//   for (unsigned int i = 0; i < classes; i ++) {
//     unsigned int chunk = i / BITS;
//     unsigned int pos = i % BITS;
//     unsigned int bit_a = encoding_a[chunk] & (1U << pos);
//     unsigned int bit_n = encoding_n[chunk] & (1U << pos);
//     unsigned int bit_p = encoding_p[chunk] & (1U << pos);
//     if ((bit_a && bit_n && bit_p) || (!bit_a && !bit_n && !bit_p)) {
//       // flip n, keep a and p
//       tm_update(encoder, i, a, bit_a, clause_output, feedback_to_clauses, feedback_to_la);
//       tm_update(encoder, i, n, !bit_n, clause_output, feedback_to_clauses, feedback_to_la);
//       tm_update(encoder, i, p, bit_p, clause_output, feedback_to_clauses, feedback_to_la);
//     } else if ((bit_a && bit_n && !bit_p) || (!bit_a && !bit_n && bit_p)) {
//       // flip a, keep n and p
//       tm_update(encoder, i, a, !bit_a, clause_output, feedback_to_clauses, feedback_to_la);
//       tm_update(encoder, i, n, bit_n, clause_output, feedback_to_clauses, feedback_to_la);
//       tm_update(encoder, i, p, bit_p, clause_output, feedback_to_clauses, feedback_to_la);
//     } else if ((bit_a && !bit_n && bit_p) || (!bit_a && bit_n && !bit_p)) {
//       // keep all
//       tm_update(encoder, i, a, bit_a, clause_output, feedback_to_clauses, feedback_to_la);
//       tm_update(encoder, i, n, bit_n, clause_output, feedback_to_clauses, feedback_to_la);
//       tm_update(encoder, i, p, bit_p, clause_output, feedback_to_clauses, feedback_to_la);
//     } else if ((bit_a && !bit_n && !bit_p) || (!bit_a && bit_n && bit_p)) {
//       // flip p, keep a and n
//       tm_update(encoder, i, a, bit_a, clause_output, feedback_to_clauses, feedback_to_la);
//       tm_update(encoder, i, n, bit_n, clause_output, feedback_to_clauses, feedback_to_la);
//       tm_update(encoder, i, p, !bit_p, clause_output, feedback_to_clauses, feedback_to_la);
//     }
//   }

// }
// static inline tsetlin_t *tk_tsetlin_alloc_encoder (lua_State *L, bool has_state)
// {
//   tsetlin_t *tm = tk_tsetlin_alloc(L);
//   tm->type = TM_ENCODER;
//   tm->has_state = has_state;
//   tm->encoder = malloc(sizeof(tsetlin_encoder_t));
//   if (!tm->encoder) luaL_error(L, "error in malloc during creation");
//   memset(tm->encoder, 0, sizeof(tsetlin_encoder_t));
//   return tm;
// }

// static inline int tk_tsetlin_init_encoder (
//   lua_State *L,
//   tsetlin_encoder_t *tm,
//   unsigned int encoding_bits,
//   unsigned int features,
//   unsigned int clauses,
//   unsigned int state_bits,
//   unsigned int threshold,
//   bool boost_true_positive,
//   double specificity
// ) {
//   tk_tsetlin_init_classifier(L, &tm->encoder,
//       encoding_bits, features, clauses, state_bits, threshold,
//       boost_true_positive, specificity);
//   tm->encoding_bits = encoding_bits;
//   tm->encoding_chunks = (encoding_bits - 1) / BITS + 1;
//   tm->encoding_filter = encoding_bits % BITS != 0
//     ? ~(((unsigned int) ~0) << (encoding_bits % BITS))
//     : (unsigned int) ~0;
//   return 0;
// }

// static inline void tk_tsetlin_create_encoder (lua_State *L)
// {
//   tsetlin_t *tm = tk_tsetlin_alloc_encoder(L, true);
//   lua_insert(L, 1);

//   tk_tsetlin_init_encoder(L, tm->encoder,
//       tk_lua_fcheckunsigned(L, 2, "hidden"),
//       tk_lua_fcheckunsigned(L, 2, "visible"),
//       tk_lua_fcheckunsigned(L, 2, "clauses"),
//       tk_lua_fcheckunsigned(L, 2, "state_bits"),
//       tk_lua_fcheckunsigned(L, 2, "target"),
//       tk_lua_fcheckboolean(L, 2, "boost_true_positive"),
//       tk_lua_fcheckposdouble(L, 2, "specificity"),
//       tk_tsetlin_get_nthreads(L, 2, "threads"));

//   lua_settop(L, 1);
// }

// static inline int tk_tsetlin_predict_encoder (lua_State *L, tsetlin_encoder_t *tm)
// {
//   lua_settop(L, 2);
//   unsigned int *bm = (unsigned int *) luaL_checkstring(L, 2);
//   unsigned int encoding_a[tm->encoding_chunks];
//   long int scores[tm->encoder.classes];
//   en_tm_encode(tm, bm, encoding_a, scores);
//   lua_pushlstring(L, (char *) encoding_a, sizeof(unsigned int) * tm->encoding_chunks);
//   lua_pushinteger(L, tm->encoder.classes);
//   return 2;
// }

// typedef struct {
//   tsetlin_encoder_t *tm;
//   unsigned int n;
//   unsigned int *next;
//   unsigned int *shuffle;
//   unsigned int *tokens;
//   double margin;
//   double loss_alpha;
//   pthread_mutex_t *qlock;
// } train_encoder_thread_data_t;

// static void *train_encoder_thread (void *arg)
// {
//   seed_rand();
//   train_encoder_thread_data_t *data = (train_encoder_thread_data_t *) arg;
//   unsigned int clause_chunks = data->tm->encoder.clause_chunks;
//   unsigned int encoder_classes = data->tm->encoder.classes;
//   unsigned int input_chunks = data->tm->encoder.input_chunks;
//   unsigned int clause_output[clause_chunks];
//   unsigned int feedback_to_clauses[clause_chunks];
//   unsigned int feedback_to_la[input_chunks];
//   long int scores[encoder_classes];
//   while (1) {
//     pthread_mutex_lock(data->qlock);
//     unsigned int next = *data->next;
//     (*data->next) ++;
//     pthread_mutex_unlock(data->qlock);
//     if (next >= data->n)
//       return NULL;
//     unsigned int idx = data->shuffle[next];
//     unsigned int *a = data->tokens + ((idx * 3 + 0) * input_chunks);
//     unsigned int *n = data->tokens + ((idx * 3 + 1) * input_chunks);
//     unsigned int *p = data->tokens + ((idx * 3 + 2) * input_chunks);
//     en_tm_update(data->tm, a, n, p,
//         clause_output, feedback_to_clauses, feedback_to_la, scores,
//         data->margin, data->loss_alpha);
//   }
//   return NULL;
// }

// static inline int tk_tsetlin_train_encoder (
//   lua_State *L,
//   tsetlin_encoder_t *tm
// ) {
//   unsigned int n = tk_lua_fcheckunsigned(L, 2, "samples");
//   unsigned int *tokens = (unsigned int *) tk_lua_fcheckstring(L, 2, "corpus");
//   double active_clause = tk_lua_fcheckposdouble(L, 2, "active_clause");
//   double margin = tk_lua_fcheckposdouble(L, 2, "margin");
//   double loss_alpha = tk_lua_fcheckposdouble(L, 2, "loss_alpha");
//   mc_tm_initialize_active_clause(&tm->encoder, active_clause);

//   pthread_t threads[n_threads];
//   pthread_mutex_t qlock;
//   pthread_mutex_init(&qlock, NULL);
//   train_encoder_thread_data_t thread_data[n_threads];

//   unsigned int next = 0;
//   unsigned int shuffle[n];
//   init_shuffle(shuffle, n);

//   for (unsigned int i = 0; i < n_threads; i++) {
//     thread_data[i].tm = tm;
//     thread_data[i].n = n;
//     thread_data[i].next = &next;
//     thread_data[i].shuffle = shuffle;
//     thread_data[i].tokens = tokens;
//     thread_data[i].margin = margin;
//     thread_data[i].loss_alpha = loss_alpha;
//     thread_data[i].qlock = &qlock;
//     if (pthread_create(&threads[i], NULL, train_encoder_thread, &thread_data[i]) != 0)
//       return tk_error(L, "pthread_create", errno);
//   }

//   // TODO: Ensure these get freed on error above
//   for (unsigned int i = 0; i < n_threads; i++)
//     if (pthread_join(threads[i], NULL) != 0)
//       return tk_error(L, "pthread_join", errno);

//   pthread_mutex_destroy(&qlock);

//   return 0;
// }

// typedef struct {
//   tsetlin_encoder_t *tm;
//   unsigned int n;
//   unsigned int *next;
//   unsigned int *tokens;
//   double margin;
//   unsigned int *correct;
//   pthread_mutex_t *lock;
//   pthread_mutex_t *qlock;
// } evaluate_encoder_thread_data_t;

// static void *evaluate_encoder_thread (void *arg)
// {
//   seed_rand();
//   evaluate_encoder_thread_data_t *data = (evaluate_encoder_thread_data_t *) arg;
//   unsigned int encoding_a[data->tm->encoding_chunks];
//   unsigned int encoding_n[data->tm->encoding_chunks];
//   unsigned int encoding_p[data->tm->encoding_chunks];
//   long int scores[data->tm->encoder.classes];
//   while (1) {
//     pthread_mutex_lock(data->qlock);
//     unsigned int next = *data->next;
//     (*data->next) += 1;
//     pthread_mutex_unlock(data->qlock);
//     if (next >= data->n)
//       return NULL;
//     unsigned int *a = data->tokens + ((next * 3 + 0) * data->tm->encoder.input_chunks);
//     unsigned int *n = data->tokens + ((next * 3 + 1) * data->tm->encoder.input_chunks);
//     unsigned int *p = data->tokens + ((next * 3 + 2) * data->tm->encoder.input_chunks);
//     en_tm_encode(data->tm, a, encoding_a, scores);
//     en_tm_encode(data->tm, n, encoding_n, scores);
//     en_tm_encode(data->tm, p, encoding_p, scores);
//     unsigned int dist_an = hamming(encoding_a, encoding_n, data->tm->encoding_bits);
//     unsigned int dist_ap = hamming(encoding_a, encoding_p, data->tm->encoding_bits);
//     if (dist_ap < dist_an) {
//       pthread_mutex_lock(data->lock);
//       (*data->correct) += 1;
//       pthread_mutex_unlock(data->lock);
//     }
//   }
//   return NULL;
// }

// static inline int tk_tsetlin_evaluate_encoder (lua_State *L, tsetlin_encoder_t *tm)
// {
//   lua_settop(L, 2);
//   unsigned int n = tk_lua_fcheckunsigned(L, 2, "samples");
//   unsigned int *tokens = (unsigned int *) tk_lua_fcheckstring(L, 2, "corpus");

//   unsigned int correct = 0;

//   pthread_t threads[n_threads];
//   pthread_mutex_t lock;
//   pthread_mutex_t qlock;
//   pthread_mutex_init(&lock, NULL);
//   pthread_mutex_init(&qlock, NULL);
//   evaluate_encoder_thread_data_t thread_data[n_threads];

//   unsigned int next = 0;

//   for (unsigned int i = 0; i < n_threads; i++) {
//     thread_data[i].tm = tm;
//     thread_data[i].n = n;
//     thread_data[i].next = &next;
//     thread_data[i].tokens = tokens;
//     thread_data[i].correct = &correct;
//     thread_data[i].lock = &lock;
//     thread_data[i].qlock = &qlock;
//     if (pthread_create(&threads[i], NULL, evaluate_encoder_thread, &thread_data[i]) != 0)
//       return tk_error(L, "pthread_create", errno);
//   }

//   // TODO: Ensure these get freed on error above
//   for (unsigned int i = 0; i < n_threads; i++)
//     if (pthread_join(threads[i], NULL) != 0)
//       return tk_error(L, "pthread_join", errno);

//   pthread_mutex_destroy(&lock);
//   pthread_mutex_destroy(&qlock);

//   lua_pushnumber(L, (double) correct / n);
//   return 1;
// }

// static inline void tk_tsetlin_persist_encoder (lua_State *L, tsetlin_encoder_t *tm, FILE *fh, bool persist_state)
// {
//   tk_lua_fwrite(L, &tm->encoding_bits, sizeof(unsigned int), 1, fh);
//   tk_lua_fwrite(L, &tm->encoding_chunks, sizeof(unsigned int), 1, fh);
//   tk_lua_fwrite(L, &tm->encoding_filter, sizeof(unsigned int), 1, fh);
//   _tk_tsetlin_persist_classifier(L, &tm->encoder, fh, persist_state);
// }

// static inline void _tk_tsetlin_load_encoder (lua_State *L, tsetlin_encoder_t *en, FILE *fh, bool read_state, bool has_state, unsigned int n_threads)
// {
//   tk_lua_fread(L, &en->encoding_bits, sizeof(unsigned int), 1, fh);
//   tk_lua_fread(L, &en->encoding_chunks, sizeof(unsigned int), 1, fh);
//   tk_lua_fread(L, &en->encoding_filter, sizeof(unsigned int), 1, fh);
//   _tk_tsetlin_load_classifier(L, &en->encoder, fh, read_state, has_state, n_threads);
// }

// static inline void tk_tsetlin_load_encoder (lua_State *L, FILE *fh, bool read_state, bool has_state)
// {
//   tsetlin_t *tm = tk_tsetlin_alloc_encoder(L, read_state);
//   unsigned int n_threads = tk_tsetlin_get_nthreads(L, 2, NULL);
//   _tk_tsetlin_load_encoder(L, en, fh, read_state, has_state, n_threads);
// }

