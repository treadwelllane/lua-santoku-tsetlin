#ifndef TK_CONF_H
#define TK_CONF_H

#include <santoku/lua/utils.h>
#include <santoku/klib.h>

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

#define _STR(x) #x
#define STR(x) _STR(x)
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

typedef uint8_t tk_bits_t;
#define BITS 8
#define BITS_DIV(x) ((x) >> 3)
#define BITS_MOD(x) ((x) & 7)
#define BYTES (BITS / CHAR_BIT)
#define BYTES_DIV(x) ((x) / BYTES)
const tk_bits_t ZERO_MASK = 0x00;
const tk_bits_t ALL_MASK = 0xFF;
const tk_bits_t POS_MASK = 0x55;
const tk_bits_t NEG_MASK = 0xAA;
static inline uint8_t popcount (tk_bits_t x) {
  return (uint8_t) __builtin_popcount(x);
}

// typedef uint64_t tk_bits_t;
// #define BITS 64
// #define BITS_DIV(x) ((x) >> 6)
// #define BITS_MOD(x) ((x) & 63)
// #define BYTES (BITS / CHAR_BIT)
// #define BYTES_DIV(x) ((x) / BYTES)
// const tk_bits_t ALL_MASK = ((tk_bits_t)0xFFFFFFFF << 32) | ((tk_bits_t)0xFFFFFFFF);
// const tk_bits_t POS_MASK = ((tk_bits_t)0x55555555 << 32) | ((tk_bits_t)0x55555555);
// const tk_bits_t NEG_MASK = ((tk_bits_t)0xAAAAAAAA << 32) | ((tk_bits_t)0xAAAAAAAA);
// static inline uint64_t popcount (tk_bits_t x) {
//   return (uint64_t) __builtin_popcountll(x);
// }

static inline uint64_t hamming_mask (
  tk_bits_t *a,
  tk_bits_t *b,
  tk_bits_t *mask,
  uint64_t chunks
) {
  uint64_t t = 0;
  for (uint64_t i = 0; i < chunks; i ++)
    t += (uint64_t) popcount((a[i] ^ b[i]) & mask[i]);
  return t;
}

static inline uint64_t hamming (
  tk_bits_t *a,
  tk_bits_t *b,
  uint64_t chunks
) {
  uint64_t t = 0;
  for (uint64_t i = 0; i < chunks; i ++)
    t += (uint64_t) popcount(a[i] ^ b[i]);
  return t;
}

static inline uint64_t mix64 (uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

static inline uint64_t hash128 (
  uint64_t lo,
  uint64_t hi
) {
  uint64_t x = lo ^ (hi + 0x9e3779b97f4a7c15ULL + (lo << 6) + (lo >> 2));
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

typedef struct { int64_t u, v; } tm_pair_t;
#define tm_pair_lt(a, b) ((a).u < (b).u || ((a).u == (b).u && (a).v < (b).v))
#define tm_pair(u, v) (((u) < (v)) ? ((tm_pair_t) { (u), (v) }) : ((tm_pair_t) { (v), (u) }))
#define tm_pair_eq(a, b) ((a).u == (b).u && ((a).v == (b).v))
#define tm_pair_hash(a) (hash128(kh_int64_hash_func((a).u), kh_int64_hash_func((a).v)))
KHASH_INIT(pairs, tm_pair_t, bool, 1, tm_pair_hash, tm_pair_eq)
typedef khash_t(pairs) tm_pairs_t;
KSORT_INIT(pair_asc, tm_pair_t, tm_pair_lt)
typedef struct { int64_t sim; bool label; } tm_dl_t;
#define tm_dl_lt(a, b) ((a).sim < (b).sim)
KSORT_INIT(dl, tm_dl_t, tm_dl_lt)
typedef kvec_t(tm_dl_t) tm_dls_t;
typedef struct { int64_t u, v; double d; } tm_candidate_t;
#define tm_candidate_lt(a, b) ((a).d < (b).d)
#define tm_candidate_gt(a, b) ((a).d > (b).d)
#define tm_candidate(u, v, d) ((tm_candidate_t) { (u), (v), (d) })
KSORT_INIT(candidates_asc, tm_candidate_t, tm_candidate_lt)
KSORT_INIT(candidates_desc, tm_candidate_t, tm_candidate_gt)
typedef kvec_t(tm_candidate_t) tm_candidates_t;

static uint64_t const multiplier = 6364136223846793005u;
__thread uint64_t mcg_state = 0xcafef00dd15ea5e5u;

// TODO: make this 64-bit
static inline uint32_t fast_rand ()
{
  uint64_t x = mcg_state;
  unsigned int count = (unsigned int) (x >> 61);
  mcg_state = x * multiplier;
  return (uint32_t) ((x ^ x >> 22) >> (22 + count));
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

static inline int fast_norm (double mean, double variance)
{
  double u1 = (double) (fast_rand() + 1) / ((double) UINT32_MAX + 1);
  double u2 = (double) fast_rand() / UINT32_MAX;
  double n1 = sqrt(-2 * log(u1)) * sin(8 * atan(1) * u2);
  return (int) round(mean + sqrt(variance) * n1);
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
    if (numa_available() == -1)
      free(p0);
    else
      numa_free(p0, s0);
    return p1;
  }
}

#endif
