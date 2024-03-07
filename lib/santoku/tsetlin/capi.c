/*

Copyright (C) 2024 Matthew Brooks (Lua integration, train/evaluate stats, drop clause)
Copyright (C) 2019 Ole-Christoffer Granmo (Original C implementation)

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
#include "roaring.c"

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TK_TSETLIN_MT "santoku_tsetlin"

struct TsetlinMachine {
  unsigned int features;
  unsigned int threshold;
  bool boost_true_positive;
  unsigned int clauses;
  unsigned int state_bits;
  unsigned int la_chunks;
  unsigned int clause_chunks;
  unsigned int filter;
	roaring_bitmap_t **ta_state;
  roaring_bitmap_t *mask1;
  roaring_bitmap_t *tmp;
	unsigned int *clause_output;
	roaring_bitmap_t *feedback_to_la;
  roaring_bitmap_t *drop_clause;
	int *feedback_to_clauses;
};

static uint64_t const multiplier = 6364136223846793005u;
static uint64_t mcg_state = 0xcafef00dd15ea5e5u;

static inline uint32_t fast_rand () {
  uint64_t x = mcg_state;
  unsigned int count = (unsigned int) (x >> 61);
  mcg_state = x * multiplier;
  return (uint32_t) ((x ^ x >> 22) >> (22 + count));
}

static inline int normal (double mean, double variance) {
  double u1 = (double) (fast_rand() + 1) / ((double) UINT32_MAX + 1);
  double u2 = (double) fast_rand() / UINT32_MAX;
  double n1 = sqrt(-2 * log(u1)) * sin(8 * atan(1) * u2);
  return (int) round(mean + sqrt(variance) * n1);
}

static inline void tm_initialize_random_streams (struct TsetlinMachine *tm, double specificity)
{
  roaring_bitmap_clear(tm->feedback_to_la);
  int n = 2 * tm->features;
  double p = 1.0 / specificity;
  int active = normal(n * p, n * p * (1 - p));
  active = active >= n ? n : active;
  active = active < 0 ? 0 : active;
  while (active --) {
    int f = fast_rand() % (2 * tm->features);
    while (roaring_bitmap_contains(tm->feedback_to_la, f))
      f = fast_rand() % (2 * tm->features);
    roaring_bitmap_add(tm->feedback_to_la, f);
  }
}

static inline void tm_inc (struct TsetlinMachine *tm, int clause, roaring_bitmap_t *actives)
{
  roaring_bitmap_t *carry = roaring_bitmap_create();
  roaring_bitmap_t *carry_next = roaring_bitmap_create();

  roaring_bitmap_clear(carry);
  roaring_bitmap_or_inplace(carry, actives);

  for (unsigned int b = 0; b < tm->state_bits; b ++)
  {
    if (roaring_bitmap_get_cardinality(carry) == 0)
      break;
    roaring_bitmap_clear(carry_next);
    roaring_bitmap_or_inplace(carry_next, tm->ta_state[clause * tm->state_bits + b]);
    roaring_bitmap_and_inplace(carry_next, carry);
    roaring_bitmap_xor_inplace(tm->ta_state[clause * tm->state_bits + b], carry);
    roaring_bitmap_clear(carry);
    roaring_bitmap_or_inplace(carry, carry_next);
  }

  if (roaring_bitmap_get_cardinality(carry) > 0)
    for (unsigned int b = 0; b < tm->state_bits; b ++)
      roaring_bitmap_or_inplace(tm->ta_state[clause * tm->state_bits + b], carry);

  roaring_bitmap_free(carry);
  roaring_bitmap_free(carry_next);
}

static inline void tm_dec (struct TsetlinMachine *tm, int clause, roaring_bitmap_t *actives)
{
  roaring_bitmap_t *carry = roaring_bitmap_create();
  roaring_bitmap_t *carry_next = roaring_bitmap_create();

  roaring_bitmap_clear(carry);
  roaring_bitmap_or_inplace(carry, actives);

  for (unsigned int b = 0; b < tm->state_bits; b ++)
  {
    if (roaring_bitmap_get_cardinality(carry) == 0)
      break;
    roaring_bitmap_clear(carry_next);
    roaring_bitmap_or_inplace(carry_next, tm->ta_state[clause * tm->state_bits + b]);
    roaring_bitmap_xor_inplace(carry_next, tm->mask1);
    roaring_bitmap_and_inplace(carry_next, carry);
    roaring_bitmap_xor_inplace(tm->ta_state[clause * tm->state_bits + b], carry);
    roaring_bitmap_clear(carry);
    roaring_bitmap_or_inplace(carry, carry_next);
  }

  if (roaring_bitmap_get_cardinality(carry) > 0)
    for (unsigned int b = 0; b < tm->state_bits; b ++)
      roaring_bitmap_andnot_inplace(tm->ta_state[clause * tm->state_bits + b], carry);

  roaring_bitmap_free(carry);
  roaring_bitmap_free(carry_next);
}

static inline int sum_up_class_votes (struct TsetlinMachine *tm, bool predict)
{
  roaring_bitmap_t *mask01 = roaring_bitmap_create();
  roaring_bitmap_t *mask10 = roaring_bitmap_create();
  roaring_bitmap_t *clause_output = roaring_bitmap_create();
  for (unsigned int i = 0; i < tm->clauses; i ++) {
    if (i % 2 == 0)
      roaring_bitmap_add(mask10, i);
    else
      roaring_bitmap_add(mask01, i);
  }
  for (unsigned int i = 0; i < tm->clauses; i ++)
    if (tm->clause_output[i / (sizeof(unsigned int) * CHAR_BIT)] & (1 << (i % (sizeof(unsigned int) * CHAR_BIT))))
      roaring_bitmap_add(clause_output, i);
  roaring_bitmap_t *tmp_pos = roaring_bitmap_create();
  roaring_bitmap_t *tmp_neg = roaring_bitmap_create();

  roaring_bitmap_clear(tmp_pos);
  roaring_bitmap_clear(tmp_neg);
  roaring_bitmap_t *drop = predict ? tm->mask1 : tm->drop_clause;
  roaring_bitmap_or_inplace(tmp_pos, clause_output);
  roaring_bitmap_or_inplace(tmp_neg, clause_output);
  roaring_bitmap_and_inplace(tmp_pos, drop);
  roaring_bitmap_and_inplace(tmp_neg, drop);
  roaring_bitmap_and_inplace(tmp_pos, mask10);
  roaring_bitmap_and_inplace(tmp_neg, mask01);
  long int class_sum =
    roaring_bitmap_get_cardinality(tmp_pos) -
    roaring_bitmap_get_cardinality(tmp_neg);
  long int threshold = tm->threshold;
	class_sum = (class_sum > threshold) ? threshold : class_sum;
	class_sum = (class_sum < -threshold) ? -threshold : class_sum;

  roaring_bitmap_free(clause_output);
  roaring_bitmap_free(mask01);
  roaring_bitmap_free(mask10);
  roaring_bitmap_free(tmp_pos);
  roaring_bitmap_free(tmp_neg);

	return class_sum;
}

static inline void tm_calculate_clause_output (struct TsetlinMachine *tm, roaring_bitmap_t *p, bool predict)
{
  memset((*tm).clause_output, 0, tm->clause_chunks * sizeof(unsigned int));
  for (long int j = 0; j < tm->clauses; j++) {
    roaring_bitmap_clear(tm->tmp);
    roaring_bitmap_t *actions = tm->ta_state[j * tm->state_bits + (tm->state_bits - 1)];
    unsigned int output = 1;
    unsigned int all_exclude = roaring_bitmap_get_cardinality(actions) == 0;
    roaring_bitmap_or_inplace(tm->tmp, actions);
    roaring_bitmap_and_inplace(tm->tmp, p);
    output = roaring_bitmap_equals(tm->tmp, actions);
    output = output && !(predict && all_exclude);
    if (output) {
      unsigned int clause_chunk = j / (sizeof(unsigned int) * CHAR_BIT);
      unsigned int clause_chunk_pos = j % (sizeof(unsigned int) * CHAR_BIT);
      (*tm).clause_output[clause_chunk] |= (1 << clause_chunk_pos);
    }
  }
}

static inline void tm_update (struct TsetlinMachine *tm, roaring_bitmap_t *bm, unsigned int target, double specificity)
{
  tm_calculate_clause_output(tm, bm, false);

  roaring_bitmap_t *feedback_to_clauses = roaring_bitmap_create();
  roaring_bitmap_t *clause_output = roaring_bitmap_create();
  for (unsigned int i = 0; i < tm->clauses; i ++)
    if (tm->clause_output[i / (sizeof(unsigned int) * CHAR_BIT)] & (1 << (i % (sizeof(unsigned int) * CHAR_BIT))))
      roaring_bitmap_add(clause_output, i);

  long int tgt = target;
  int class_sum = sum_up_class_votes(tm, false);
  double p = (1.0/(tm->threshold*2))*(tm->threshold + (1 - 2*tgt)*class_sum);

  roaring_bitmap_clear(feedback_to_clauses);
  for (unsigned int i = 0; i < tm->clauses; i ++)
    if (((float)fast_rand())/((float)UINT32_MAX) <= p)
      roaring_bitmap_add(feedback_to_clauses, i);
  roaring_bitmap_and_inplace(feedback_to_clauses, tm->drop_clause);

	for (long int j = 0; j < tm->clauses; j++) {
		if (!roaring_bitmap_contains(feedback_to_clauses, j))
			continue;
		if ((2 * tgt - 1) * (1 - 2 * (j & 1)) == -1) {

      // Type II feedback
			if (roaring_bitmap_contains(clause_output, j))
      {
        roaring_bitmap_clear(tm->tmp);
        roaring_bitmap_or_inplace(tm->tmp, bm);
        roaring_bitmap_xor_inplace(tm->tmp, tm->mask1);
        roaring_bitmap_andnot_inplace(tm->tmp, tm->ta_state[j * tm->state_bits + (tm->state_bits - 1)]);
        tm_inc(tm, j, tm->tmp);
      }

		} else if ((2*tgt-1) * (1 - 2 * (j & 1)) == 1) {

			// Type I Feedback
			tm_initialize_random_streams(tm, specificity);

			if (roaring_bitmap_contains(clause_output, j)) {
        if (tm->boost_true_positive) {
          tm_inc(tm, j, bm);
        } else {
          roaring_bitmap_clear(tm->tmp);
          roaring_bitmap_or_inplace(tm->tmp, bm);
          roaring_bitmap_andnot_inplace(tm->tmp, tm->feedback_to_la);
          tm_inc(tm, j, tm->tmp);
        }
        roaring_bitmap_clear(tm->tmp);
        roaring_bitmap_or_inplace(tm->tmp, bm);
        roaring_bitmap_xor_inplace(tm->tmp, tm->mask1);
        roaring_bitmap_and_inplace(tm->tmp, tm->feedback_to_la);
        tm_dec(tm, j, tm->tmp);
			} else {
        tm_dec(tm, j, tm->feedback_to_la);
			}

		}
	}

  roaring_bitmap_free(feedback_to_clauses);
  roaring_bitmap_free(clause_output);
}

static inline int tm_score (struct TsetlinMachine *tm, roaring_bitmap_t *bm) {
	tm_calculate_clause_output(tm, bm, true);
	return sum_up_class_votes(tm, true);
}

struct MultiClassTsetlinMachine {
  unsigned int classes;
  unsigned int features;
  unsigned int la_chunks;
  unsigned int clauses;
  unsigned int clause_chunks;
  unsigned int state_bits;
  roaring_bitmap_t *drop_clause;
  roaring_bitmap_t *mask1;
  roaring_bitmap_t *tmp;
	struct TsetlinMachine **tsetlin_machines;
};

static inline void mc_tm_initialize (struct MultiClassTsetlinMachine *mc_tm)
{
	for (long int i = 0; i < mc_tm->classes; i++) {
    struct TsetlinMachine *tm = mc_tm->tsetlin_machines[i];
    for (long int j = 0; j < tm->clauses; ++j) {
      for (long int b = 0; b < tm->state_bits - 1; ++b)
      {
        roaring_bitmap_clear(tm->ta_state[j * tm->state_bits + b]);
        roaring_bitmap_or_inplace(tm->ta_state[j * tm->state_bits + b], tm->mask1);
      }
      roaring_bitmap_clear(tm->ta_state[j * tm->state_bits + (tm->state_bits - 1)]);
    }
  }
}

static inline unsigned int mc_tm_predict (struct MultiClassTsetlinMachine *tm, roaring_bitmap_t *bm)
{
	long int max_class = 0;
	long int max_class_sum = tm_score(tm->tsetlin_machines[0], bm);
  for (long int i = 1; i < tm->classes; i++) {
    int class_sum = tm_score(tm->tsetlin_machines[i], bm);
    if (max_class_sum < class_sum) {
      max_class_sum = class_sum;
      max_class = i;
    }
  }
	return max_class;
}

static inline void mc_tm_update (struct MultiClassTsetlinMachine *tm, roaring_bitmap_t *bm, unsigned int target_class, double specificity)
{
	tm_update(tm->tsetlin_machines[target_class], bm, 1, specificity);
	unsigned int negative_target_class =
    ((unsigned int) tm->classes * 1.0 * rand()) /
    ((unsigned int) RAND_MAX + 1);
	while (negative_target_class == target_class)
		negative_target_class = (unsigned int)tm->classes * 1.0*rand()/((unsigned int)RAND_MAX + 1);
	tm_update(tm->tsetlin_machines[negative_target_class], bm, 0, specificity);
}

static inline void mc_tm_initialize_drop_clause (struct MultiClassTsetlinMachine *tm, double drop_clause)
{
  roaring_bitmap_clear(tm->drop_clause);
  for (unsigned int i = 0; i < tm->clauses; i ++)
    if (((float)fast_rand())/((float)UINT32_MAX) <= drop_clause)
      roaring_bitmap_add(tm->drop_clause, i);
}

struct MultiClassTsetlinMachine **tk_tsetlin_peekp (lua_State *L, int i)
{
  return (struct MultiClassTsetlinMachine **) luaL_checkudata(L, i, TK_TSETLIN_MT);
}

struct MultiClassTsetlinMachine *tk_tsetlin_peek (lua_State *L, int i)
{
  return *tk_tsetlin_peekp(L, i);
}

static inline unsigned int tk_tsetlin_checkunsigned (lua_State *L, int i)
{
  lua_Integer l = luaL_checkinteger(L, i);
  if (l < 0)
    luaL_error(L, "value can't be negative");
  if (l > UINT_MAX)
    luaL_error(L, "value is too large");
  return (unsigned int) l;
}

static inline void tk_tsetlin_register (lua_State *L, luaL_Reg *regs, int nup)
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

static inline int tk_tsetlin_create (lua_State *L)
{
  lua_settop(L, 6);
	struct MultiClassTsetlinMachine *mc_tm;
	mc_tm = (void *)malloc(sizeof(struct MultiClassTsetlinMachine));
  mc_tm->classes = tk_tsetlin_checkunsigned(L, 1);
  mc_tm->features = tk_tsetlin_checkunsigned(L, 2);
  mc_tm->clauses = tk_tsetlin_checkunsigned(L, 3);
  mc_tm->la_chunks = (2 * mc_tm->features - 1) / (sizeof(unsigned int) * CHAR_BIT) + 1;
  mc_tm->clause_chunks = (mc_tm->clauses - 1) / (sizeof(unsigned int) * CHAR_BIT) + 1;
  mc_tm->drop_clause = roaring_bitmap_create();
  mc_tm->tsetlin_machines = malloc(sizeof(struct TsetlinMachine *) * mc_tm->classes);
  mc_tm->state_bits = tk_tsetlin_checkunsigned(L, 4);
  mc_tm->mask1 = roaring_bitmap_create();
  mc_tm->tmp = roaring_bitmap_create();
  roaring_bitmap_add_range(mc_tm->mask1, 0, mc_tm->features * 2);
	for (long int i = 0; i < mc_tm->classes; i++) {
    struct TsetlinMachine *tm = (void *)malloc(sizeof(struct TsetlinMachine));
		mc_tm->tsetlin_machines[i] = tm;
    tm->features = mc_tm->features;
    tm->clauses = mc_tm->clauses;
    tm->state_bits = mc_tm->state_bits;
    tm->threshold = tk_tsetlin_checkunsigned(L, 5);
    luaL_checktype(L, 6, LUA_TBOOLEAN);
    tm->boost_true_positive = lua_toboolean(L, 6);
    tm->la_chunks = mc_tm->la_chunks;
    tm->clause_chunks = mc_tm->clause_chunks;
    tm->drop_clause = mc_tm->drop_clause;
    tm->filter = (tm->features * 2) % (sizeof(unsigned int) * CHAR_BIT) != 0
      ? ~(((unsigned int) ~0) << ((tm->features * 2) % (sizeof(unsigned int) * CHAR_BIT)))
      : (unsigned int) ~0;
    tm->clause_output = malloc(sizeof(unsigned int) * tm->clause_chunks);
    tm->feedback_to_la = roaring_bitmap_create();
    tm->feedback_to_clauses = malloc(sizeof(int) * tm->clause_chunks);
    tm->tmp = mc_tm->tmp;
    tm->mask1 = mc_tm->mask1;
    tm->ta_state = malloc(sizeof(roaring_bitmap_t *) * tm->clauses * tm->state_bits);
    for (unsigned int i = 0; i < tm->clauses * tm->state_bits; i ++)
      tm->ta_state[i] = roaring_bitmap_create();
  }
  mc_tm_initialize(mc_tm);
  struct MultiClassTsetlinMachine **mc_tmp = (struct MultiClassTsetlinMachine **)
    lua_newuserdata(L, sizeof(struct MultiClassTsetlinMachine *));
  *mc_tmp = mc_tm;
  luaL_getmetatable(L, TK_TSETLIN_MT);
  lua_setmetatable(L, -2);
  return 1;
}

static inline int tk_tsetlin_destroy (lua_State *L)
{
  lua_settop(L, 1);
  struct MultiClassTsetlinMachine **mc_tmp = tk_tsetlin_peekp(L, 1);
  struct MultiClassTsetlinMachine *mc_tm = *mc_tmp;
  if (mc_tm == NULL)
    return 0;
	for (long int i = 0; i < mc_tm->classes; i++) {
    free(mc_tm->tsetlin_machines[i]->ta_state);
    free(mc_tm->tsetlin_machines[i]->clause_output);
    free(mc_tm->tsetlin_machines[i]->feedback_to_clauses);
    roaring_bitmap_free(mc_tm->tsetlin_machines[i]->feedback_to_la);
    for (unsigned int i = 0; i < mc_tm->clauses * mc_tm->state_bits; i ++)
      roaring_bitmap_free(mc_tm->tsetlin_machines[i]->ta_state[i]);
    free(mc_tm->tsetlin_machines[i]->ta_state);
    free(mc_tm->tsetlin_machines[i]);
  }
  roaring_bitmap_free(mc_tm->mask1);
  roaring_bitmap_free(mc_tm->tmp);
  free(mc_tm->tsetlin_machines);
  free(mc_tm);
  *mc_tmp = NULL;
  return 0;
}

static inline int tk_tsetlin_predict (lua_State *L)
{
  lua_settop(L, 3);
  struct MultiClassTsetlinMachine *tm = tk_tsetlin_peek(L, 1);
  roaring_bitmap_t *bm = *((roaring_bitmap_t **) luaL_checkudata(L, 2, "santoku_bitmap"));
  unsigned int class = mc_tm_predict(tm,  bm);
  lua_pushinteger(L, class);
  return 1;
}

static inline int tk_tsetlin_update (lua_State *L)
{
  lua_settop(L, 5);
  struct MultiClassTsetlinMachine *tm = tk_tsetlin_peek(L, 1);
  roaring_bitmap_t *bm = *((roaring_bitmap_t **) luaL_checkudata(L, 2, "santoku_bitmap")); // i p
  lua_Integer tgt = luaL_checkinteger(L, 3);
  if (tgt < 0)
    luaL_error(L, "target class must be greater than zero");
  double specificity = luaL_checknumber(L, 4);
  double drop_clause = luaL_optnumber(L, 5, 1);
  mc_tm_initialize_drop_clause(tm, drop_clause);
  mc_tm_update(tm, bm, tgt, specificity);
  return 0;
}

static inline int tk_tsetlin_train (lua_State *L)
{
  lua_settop(L, 5);
  struct MultiClassTsetlinMachine *tm = tk_tsetlin_peek(L, 1);
  unsigned int i_ps = 2; luaL_checktype(L, i_ps, LUA_TTABLE);
  unsigned int i_ss = 3; luaL_checktype(L, i_ss, LUA_TTABLE);
  unsigned int n = lua_objlen(L, i_ps);
  double specificity = luaL_checknumber(L, 4);
  double drop_clause = luaL_optnumber(L, 5, 1);
  mc_tm_initialize_drop_clause(tm, drop_clause);
  for (unsigned int i = 1; i <= n; i ++) {
    lua_pushinteger(L, i); // i
    lua_gettable(L, i_ps); // p
    roaring_bitmap_t *bm = *((roaring_bitmap_t **) luaL_checkudata(L, -1, "santoku_bitmap")); // i p
    lua_pushinteger(L, i); // p i
    lua_gettable(L, i_ss); // p s
    lua_Integer s = luaL_checkinteger(L, -1);
    mc_tm_update(tm, bm, s, specificity);
    lua_pop(L, 2);
  }
  return 0;
}

static inline int tk_tsetlin_evaluate (lua_State *L)
{
  lua_settop(L, 4);
  struct MultiClassTsetlinMachine *tm = tk_tsetlin_peek(L, 1);
  unsigned int i_ps = 2; luaL_checktype(L, i_ps, LUA_TTABLE);
  unsigned int i_ss = 3; luaL_checktype(L, i_ss, LUA_TTABLE);
  unsigned int n = lua_objlen(L, i_ps);
  bool track_stats = lua_toboolean(L, 4);
  unsigned int correct = 0;
  unsigned int *confusion;
  unsigned int *predictions;
  unsigned int *observations;
  if (track_stats) {
    confusion = malloc(sizeof(unsigned int) * tm->classes * tm->classes);
    predictions = malloc(sizeof(unsigned int) * tm->classes);
    observations = malloc(sizeof(unsigned int) * tm->classes);
    if (!(confusion && predictions && observations))
      luaL_error(L, "error in malloc during evaluation");
    memset(confusion, 0, sizeof(unsigned int) * tm->classes * tm->classes);
    memset(predictions, 0, sizeof(unsigned int) * tm->classes);
    memset(observations, 0, sizeof(unsigned int) * tm->classes);
  }
  for (unsigned int i = 1; i <= n; i ++) {
    lua_pushinteger(L, i); // i
    lua_gettable(L, i_ss); // s
    unsigned int expected = luaL_checkinteger(L, -1);
    lua_pushinteger(L, i); // s i
    lua_gettable(L, i_ps); // s p
    roaring_bitmap_t *p = *((roaring_bitmap_t **) luaL_checkudata(L, -1, "santoku_bitmap"));
    unsigned int predicted = mc_tm_predict(tm, p);
    if (expected == predicted)
      correct ++;
    if (track_stats) {
      observations[expected] ++;
      predictions[predicted] ++;
      if (expected != predicted)
        confusion[expected * tm->classes + predicted] ++;
    }
    lua_pop(L, 2); //
  }
  lua_pushnumber(L, correct);
  if (track_stats) {
    lua_newtable(L); // ct
    for (unsigned int i = 0; i < tm->classes; i ++) {
      for (unsigned int j = 0; j < tm->classes; j ++) {
        unsigned int c = confusion[i * tm->classes + j];
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
    for (unsigned int i = 0; i < tm->classes; i ++) {
      lua_pushinteger(L, i); // pt i
      lua_pushinteger(L, predictions[i]); // pt i p
      lua_settable(L, -3); // pt
    }
    lua_newtable(L); // ot
    for (unsigned int i = 0; i < tm->classes; i ++) {
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

static luaL_Reg tk_tsetlin_fns[] =
{
  { "create", tk_tsetlin_create },
  { "destroy", tk_tsetlin_destroy },
  { "update", tk_tsetlin_update },
  { "predict", tk_tsetlin_predict },
  { "train", tk_tsetlin_train },
  { "evaluate", tk_tsetlin_evaluate },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_capi (lua_State *L)
{
  lua_newtable(L); // t
  tk_tsetlin_register(L, tk_tsetlin_fns, 0); // t
  luaL_newmetatable(L, TK_TSETLIN_MT); // t mt
  lua_pushcfunction(L, tk_tsetlin_destroy); // t mt fn
  lua_setfield(L, -2, "__gc"); // t mt
  lua_pop(L, 1); // t
  return 1;
}
