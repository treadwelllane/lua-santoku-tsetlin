#ifndef TK_INV_H
#define TK_INV_H

#include <assert.h>
#include <santoku/lua/utils.h>
#include <santoku/tsetlin/conf.h>
#include <santoku/ivec.h>
#include <santoku/iumap.h>
#include <santoku/threads.h>

#define TK_INV_MT "tk_inv_t"
#define TK_INV_EPH "tk_inv_eph"

typedef tk_ivec_t * tk_inv_posting_t;
#define tk_vec_name tk_inv_postings
#define tk_vec_base tk_inv_posting_t
#define tk_vec_pushbase(L, x) tk_lua_get_ephemeron(L, TK_INV_EPH, x)
#define tk_vec_peekbase(L, i) tk_ivec_peek(L, i, "posting")
#define tk_vec_limited
#include <santoku/vec/tpl.h>

typedef tk_rvec_t * tk_inv_hood_t;
#define tk_vec_name tk_inv_hoods
#define tk_vec_base tk_inv_hood_t
#define tk_vec_pushbase(L, x) tk_lua_get_ephemeron(L, TK_INV_EPH, x)
#define tk_vec_peekbase(L, i) tk_rvec_peek(L, i, "hood")
#define tk_vec_limited
#include <santoku/vec/tpl.h>

#define TK_INV_LENGTH_BUMP 1e-9
#define TK_INV_LENGTH_EPS 5e-10

static inline double tk_inv_length_bump (double v, size_t l) {
  return l > 0 ? v + TK_INV_LENGTH_BUMP * (1.0 / (double) (l + 1)) : v;
}

static inline double tk_inv_length_unbump (double v, size_t l) {
  v = l > 0 ? v - TK_INV_LENGTH_BUMP * (1.0 / (double) (l + 1)) : v;
  return fabs(v) < TK_INV_LENGTH_EPS ? 0 : v;
}

typedef enum {
  TK_INV_JACCARD,
  TK_INV_OVERLAP,
} tk_inv_cmp_type_t;

typedef enum {
  TK_INV_NEIGHBORHOODS,
} tk_inv_stage_t;

typedef struct tk_inv_thread_s tk_inv_thread_t;

typedef struct tk_inv_s {
  bool destroyed;
  int64_t next_sid;
  uint64_t features;
  tk_iumap_t *uid_sid;
  tk_iumap_t *sid_uid;
  tk_ivec_t *node_offsets;
  tk_ivec_t *node_bits;
  tk_inv_postings_t *postings;
  tk_inv_thread_t *threads;
  tk_threadpool_t *pool;
} tk_inv_t;

typedef struct tk_inv_thread_s {
  tk_inv_t *I;
  tk_inv_hoods_t *hoods;
  tk_iuset_t *seen;
  tk_ivec_t *cnt;
  tk_ivec_t *touched;
  tk_iumap_t *sid_idx;
  tk_ivec_t *uids;
  tk_ivec_t *sids;
  uint64_t ifirst, ilast;
  double eps;
  uint64_t knn;
  tk_inv_cmp_type_t cmp;
} tk_inv_thread_t;

static inline tk_inv_t *tk_inv_peek (lua_State *L, int i)
{
  return (tk_inv_t *) luaL_checkudata(L, i, TK_INV_MT);
}

static inline tk_inv_t *tk_inv_peekopt (lua_State *L, int i)
{
  return (tk_inv_t *) tk_lua_testuserdata(L, i, TK_INV_MT);
}

static inline void tk_inv_shrink (
  tk_inv_t *I
) {
  if (I->destroyed)
    return;
  #warning todo
}

static inline void tk_inv_destroy (
  tk_inv_t *I
) {
  if (I->destroyed)
    return;
  I->destroyed = true;
  tk_iumap_destroy(I->uid_sid);
  tk_iumap_destroy(I->sid_uid);
  for (uint64_t i = 0; i < I->pool->n_threads; i ++) {
    tk_inv_thread_t *data = I->threads + i;
    tk_iuset_destroy(data->seen);
  }
  tk_threads_destroy(I->pool);
  free(I->threads);
}

static inline double overlap (
  int64_t *a, size_t alen,
  int64_t *b, size_t blen
) {
  if (alen == 0 || blen == 0)
    return 0.0;
  size_t i = 0, j = 0, inter = 0;
  while (i < alen && j < blen) {
    if (a[i] == b[j]) { inter++; i++; j++; }
    else if (a[i] < b[j]) i++;
    else j++;
  }
  size_t min_len = (alen < blen) ? alen : blen;
  return (double) inter / (double) min_len;
}

static inline double tk_inv_jaccard (
  int64_t *abits, size_t asize,
  int64_t *bbits, size_t bsize
) {
  size_t i = 0, j = 0;
  size_t intersection = 0;
  size_t union_count = 0;
  while (i < asize && j < bsize) {
    if (abits[i] == bbits[j]) {
      intersection ++;
      union_count ++;
      i ++;
      j ++;
    } else if (abits[i] < bbits[j]) {
      union_count ++;
      i ++;
    } else {
      union_count ++;
      j ++;
    }
  }
  // Remaining elements
  union_count += (asize - i) + (bsize - j);
  if (union_count == 0) // both empty → define Jaccard as 1.0
    return 1.0;
  return (double) intersection / (double) union_count;
}

static inline double tk_inv_similarity (
  size_t inter,
  size_t qlen,
  size_t elen,
  tk_inv_cmp_type_t cmp
) {
  switch (cmp) {
    case TK_INV_JACCARD: {
      size_t uni = qlen + elen - inter;
      return (uni == 0) ? 0.0 : (double) inter / (double) uni;
    }
    case TK_INV_OVERLAP: {
      size_t min_len = (qlen < elen) ? qlen : elen;
      return (min_len == 0) ? 0.0 : (double) inter / (double) min_len;
    }
    default:
      return tk_inv_similarity(inter, qlen, elen, TK_INV_JACCARD);
  }
}

static inline void tk_inv_persist (
  lua_State *L,
  tk_inv_t *I,
  FILE *fh
) {
  if (I->destroyed) {
    tk_lua_verror(L, 2, "persist", "can't persist a destroyed index");
    return;
  }
  // core scalars
  tk_lua_fwrite(L, (char *) &I->destroyed, sizeof(bool), 1, fh);
  tk_lua_fwrite(L, (char *) &I->next_sid, sizeof(int64_t),1, fh);
  tk_lua_fwrite(L, (char *) &I->features, sizeof(uint64_t), 1, fh);
  // uid → sid map
  khint_t m = I->uid_sid ? tk_iumap_size(I->uid_sid) : 0;
  tk_lua_fwrite(L, (char *) &m, sizeof(khint_t), 1, fh);
  if (m)
    for (khint_t i = tk_iumap_begin(I->uid_sid); i < tk_iumap_end(I->uid_sid); i ++)
      if (tk_iumap_exist(I->uid_sid, i)) {
        int64_t k = (int64_t) tk_iumap_key(I->uid_sid, i);
        int64_t v = (int64_t) tk_iumap_value(I->uid_sid, i);
        tk_lua_fwrite(L, (char *) &k, sizeof(int64_t), 1, fh);
        tk_lua_fwrite(L, (char *) &v, sizeof(int64_t), 1, fh);
      }
  // sid → uid map
  m = I->sid_uid ? tk_iumap_size(I->sid_uid) : 0;
  tk_lua_fwrite(L, (char *) &m, sizeof(khint_t), 1, fh);
  if (m)
    for (khint_t i = tk_iumap_begin(I->sid_uid); i < tk_iumap_end(I->sid_uid); i ++)
      if (tk_iumap_exist(I->sid_uid, i)) {
        int64_t k = (int64_t) tk_iumap_key(I->sid_uid, i);
        int64_t v = (int64_t) tk_iumap_value(I->sid_uid, i);
        tk_lua_fwrite(L, (char *) &k,sizeof(int64_t), 1, fh);
        tk_lua_fwrite(L, (char *) &v,sizeof(int64_t), 1, fh);
      }
  // node_offsets
  uint64_t n = I->node_offsets ? I->node_offsets->n : 0;
  tk_lua_fwrite(L,(char *) &n, sizeof(uint64_t), 1, fh);
  if (n)
    tk_lua_fwrite(L, (char *) I->node_offsets->a, sizeof(int64_t), n, fh);
  // node_bits
  n = I->node_bits ? I->node_bits->n : 0;
  tk_lua_fwrite(L,(char *) &n, sizeof(uint64_t), 1, fh);
  if (n)
    tk_lua_fwrite(L, (char *) I->node_bits->a, sizeof(int64_t), n, fh);
  // postings: vector of posting lists
  uint64_t pcount = I->postings ? I->postings->n : 0;
  tk_lua_fwrite(L, (char *) &pcount, sizeof(uint64_t), 1, fh);
  for (uint64_t i = 0; i < pcount; i ++) {
    tk_inv_posting_t P = I->postings->a[i];
    tk_lua_fwrite(L, (char *) &P->n, sizeof(uint64_t), 1, fh);
    tk_lua_fwrite(L, (char *) P->a, sizeof(int64_t), P->n, fh);
  }
}

static inline uint64_t tk_inv_size (
  tk_inv_t *I
) {
  return tk_iumap_size(I->uid_sid);
}

static inline void tk_inv_uid_remove (
  tk_inv_t *I,
  int64_t uid
) {
  khint_t khi;
  khi = tk_iumap_get(I->uid_sid, uid);
  if (khi == tk_iumap_end(I->uid_sid))
    return;
  int64_t sid = tk_iumap_value(I->uid_sid, khi);
  tk_iumap_del(I->uid_sid, khi);
  khi = tk_iumap_get(I->sid_uid, sid);
  if (khi == tk_iumap_end(I->sid_uid))
    return;
  tk_iumap_del(I->sid_uid, khi);
}

static inline int64_t tk_inv_uid_sid (
  tk_inv_t *I,
  int64_t uid,
  bool create
) {
  int kha;
  khint_t khi;
  if (create) {
    int64_t sid = (int64_t) (I->next_sid ++);
    khi = tk_iumap_put(I->uid_sid, uid, &kha);
    if (!kha) {
      int64_t sid0 = tk_iumap_value(I->uid_sid, khi);
      khi = tk_iumap_get(I->sid_uid, sid0);
      if (khi != tk_iumap_end(I->sid_uid))
        tk_iumap_del(I->sid_uid, khi);
    }
    tk_iumap_value(I->uid_sid, khi) = sid;
    khi = tk_iumap_put(I->sid_uid, sid, &kha);
    tk_iumap_value(I->sid_uid, khi) = uid;
    return sid;
  } else {
    khi = tk_iumap_get(I->uid_sid, uid);
    if (khi == tk_iumap_end(I->uid_sid))
      return -1;
    else
      return tk_iumap_value(I->uid_sid, khi);
  }
}

static inline int64_t tk_inv_sid_uid (
  tk_inv_t *I,
  int64_t sid
) {
  khint_t khi = tk_iumap_get(I->sid_uid, sid);
  if (khi == tk_iumap_end(I->sid_uid))
    return -1;
  else
    return tk_iumap_value(I->sid_uid, khi);
}

static inline int64_t *tk_inv_sget (
  tk_inv_t *I,
  int64_t sid,
  size_t *np
) {
  int64_t start = I->node_offsets->a[sid];
  int64_t end = sid + 1 == (int64_t) I->node_offsets->n ? (int64_t) I->node_bits->n : I->node_offsets->a[sid + 1];
  *np = (size_t) (end - start);
  return I->node_bits->a + start;
}

static inline int64_t *tk_inv_get (
  tk_inv_t *I,
  int64_t uid,
  size_t *np
) {
  int64_t sid = tk_inv_uid_sid(I, uid, false);
  if (sid < 0)
    return NULL;
  return tk_inv_sget(I, sid, np);
}

static inline void tk_inv_add (
  lua_State *L,
  tk_inv_t *I,
  int Ii,
  tk_ivec_t *ids,
  tk_ivec_t *node_bits
) {
  if (I->destroyed) {
    tk_lua_verror(L, 2, "add", "can't add to a destroyed index");
    return;
  }
  if (node_bits->n == 0)
    return;
  int kha;
  khint_t khi;
  tk_iuset_t *seen = tk_iuset_create();
  for (uint64_t i = 0; i < node_bits->n; i ++) {
    int64_t b = node_bits->a[i];
    if (b < 0)
      continue;
    int64_t s = b / (int64_t) I->features;
    if (s >= (int64_t) ids->n)
      continue;
    int64_t fid = b % (int64_t) I->features;
    int64_t uid = ids->a[s];
    khi = tk_iuset_put(seen, uid, &kha);
    int64_t sid = tk_inv_uid_sid(I, uid, kha);
    if (kha)
      tk_ivec_push(I->node_offsets, (int64_t) I->node_bits->n);
    tk_ivec_push(I->postings->a[fid], sid);
    tk_ivec_push(I->node_bits, fid);
  }
  tk_iuset_destroy(seen);
}

static inline void tk_inv_remove (
  lua_State *L,
  tk_inv_t *I,
  int64_t uid
) {
  if (I->destroyed) {
    tk_lua_verror(L, 2, "remove", "can't remove from a destroyed index");
    return;
  }
  tk_inv_uid_remove(I, uid);
}

static inline void tk_inv_neighborhoods (
  lua_State *L,
  tk_inv_t *I,
  uint64_t knn,
  double eps,
  tk_inv_cmp_type_t cmp,
  tk_inv_hoods_t **hoodsp,
  tk_ivec_t **uidsp
) {
  if (I->destroyed) {
    tk_lua_verror(L, 2, "neighborhoods", "can't query a destroyed index");
    return;
  }
  int kha;
  khint_t khi;
  tk_ivec_t *sids, *uids;
  if (uidsp && *uidsp) {
    sids = tk_ivec_create(L, (*uidsp)->n, 0, 0);
    uids = tk_ivec_create(L, (*uidsp)->n, 0, 0);
    tk_ivec_copy(uids, *uidsp, 0, (int64_t) (*uidsp)->n, 0);
    for (uint64_t i = 0; i < uids->n; i ++)
      sids->a[i] = tk_inv_uid_sid(I, uids->a[i], false);
  } else {
    sids = tk_iumap_values(L, I->uid_sid);
    tk_ivec_asc(sids, 0, sids->n); // sort for cache locality
    uids = tk_ivec_create(L, sids->n, 0, 0);
    for (uint64_t i = 0; i < sids->n; i ++)
      uids->a[i] = tk_inv_sid_uid(I, sids->a[i]);
  }
  tk_iumap_t *sid_idx = tk_iumap_create();
  for (uint64_t i = 0; i < sids->n; i ++) {
    khi = tk_iumap_put(sid_idx, sids->a[i], &kha);
    tk_iumap_value(sid_idx, khi) = (int64_t) i;
  }
  tk_inv_hoods_t *hoods = tk_inv_hoods_create(L, uids->n, 0, 0);
  for (uint64_t i = 0; i < hoods->n; i ++) {
    hoods->a[i] = tk_rvec_create(L, knn, 0, 0);
    hoods->a[i]->n = 0;
    tk_lua_add_ephemeron(L, TK_INV_EPH, -2, -1);
    lua_pop(L, 1);
  }
  for (uint64_t i = 0; i < I->pool->n_threads; i ++) {
    tk_inv_thread_t *data = I->threads + i;
    data->uids = uids;
    data->sids = sids;
    tk_ivec_ensure(data->cnt, sids->n);
    data->hoods = hoods;
    data->sid_idx = sid_idx;
    data->eps = eps + TK_INV_LENGTH_EPS;
    data->knn = knn;
    data->cmp = cmp;
    tk_thread_range(i, I->pool->n_threads, hoods->n, &data->ifirst, &data->ilast);
  }
  tk_threads_signal(I->pool, TK_INV_NEIGHBORHOODS, 0);
  tk_iumap_destroy(sid_idx);
  if (hoodsp) *hoodsp = hoods;
  if (uidsp) *uidsp = uids;
  lua_remove(L, -3); // sids
}

static inline tk_rvec_t *tk_inv_neighbors_by_vec (
  tk_inv_t *I,
  int64_t *data,
  size_t datalen,
  int64_t sid0,
  uint64_t knn,
  double eps,
  tk_rvec_t *out,
  tk_inv_cmp_type_t cmp
) {
  eps += TK_INV_LENGTH_EPS;
  tk_rvec_ensure(out, knn);
  if (knn)
    out->m = knn;
  if (datalen == 0)
    return out;
  size_t n_sids = I->node_offsets->n;
  tk_ivec_t *cnt = tk_ivec_create(NULL, n_sids, 0, 0);
  tk_ivec_zero(cnt);
  tk_ivec_t *touched = tk_ivec_create(NULL, 0, 0, 0);
  for (size_t i = 0; i < datalen; i++) {
    int64_t fid = data[i];
    if (fid < 0 || fid >= (int64_t)I->postings->n)
      continue;
    tk_ivec_t *vsids = I->postings->a[fid];
    for (uint64_t j = 0; j < vsids->n; j++) {
      int64_t vsid = vsids->a[j];
      if (vsid == sid0)
        continue;
      if (cnt->a[vsid] ++ == 0)
        tk_ivec_push(touched, vsid);
    }
  }
  for (uint64_t i = 0; i < touched->n; i ++) {
    int64_t vsid = touched->a[i];
    uint64_t inter = (uint64_t) cnt->a[vsid];
    size_t elen;
    tk_inv_sget(I, vsid, &elen);
    double sim  = tk_inv_similarity(inter, datalen, elen, cmp);
    double dist = tk_inv_length_bump(1.0 - sim, elen);
    if (dist <= eps) {
      int64_t vuid = tk_inv_sid_uid(I, vsid);
      if (vuid >= 0) {
        if (knn)
          tk_rvec_hmax(out, tk_rank(vuid, dist));
        else
          tk_rvec_push(out, tk_rank(vuid, dist));
      }
    }
    cnt->a[vsid] = 0;
  }
  tk_rvec_asc(out, 0, out->n);
  for (uint64_t i = 0; i < out->n; ++i) {
    size_t len = 0;
    tk_inv_sget(I, tk_inv_uid_sid(I, out->a[i].i, false), &len);
    out->a[i].d = tk_inv_length_unbump(out->a[i].d, len);
  }
  if (knn > 0 && out->n > knn)
    out->n = knn;
  tk_ivec_destroy(cnt);
  tk_ivec_destroy(touched);
  return out;
}

static inline tk_rvec_t *tk_inv_neighbors_by_id (
  tk_inv_t *I,
  int64_t uid,
  uint64_t knn,
  double eps,
  tk_rvec_t *out,
  tk_inv_cmp_type_t cmp
) {
  int64_t sid0 = tk_inv_uid_sid(I, uid, false);
  tk_rvec_clear(out);
  if (sid0 < 0)
    return out;
  size_t len = 0;
  int64_t *data = tk_inv_get(I, uid, &len);
  return tk_inv_neighbors_by_vec(I, data, len, sid0, knn, eps, out, cmp);
}

static inline int tk_inv_gc_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  tk_inv_destroy(I);
  return 0;
}

static inline int tk_inv_add_lua (lua_State *L)
{
  int Ii = 1;
  tk_inv_t *I = tk_inv_peek(L, 1);
  tk_ivec_t *node_bits = tk_ivec_peek(L, 2, "node_bits");
  if (lua_type(L, 3) == LUA_TNUMBER) {
    int64_t s = (int64_t) tk_lua_checkunsigned(L, 3, "base_id");
    uint64_t n = tk_lua_checkunsigned(L, 4, "n_nodes");
    tk_ivec_t *ids = tk_ivec_create(L, n, 0, 0);
    tk_ivec_fill_indices(ids);
    tk_ivec_add(ids, s, 0, ids->n);
    tk_inv_add(L, I, Ii, ids, node_bits);
    lua_pop(L, 1);
  } else {
    tk_inv_add(L, I, Ii, tk_ivec_peek(L, 3, "ids"), node_bits);
  }
  return 0;
}

static inline int tk_inv_remove_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  int64_t id = tk_lua_checkinteger(L, 2, "id");
  tk_inv_remove(L, I, id);
  return 0;
}

static inline int tk_inv_get_lua (lua_State *L)
{
  lua_settop(L, 3);
  tk_inv_t *I = tk_inv_peek(L, 1);
  int64_t uid = -1;
  tk_ivec_t *uids = NULL;
  tk_ivec_t *out = tk_ivec_peekopt(L, 3);
  out = out == NULL ? tk_ivec_create(L, 0, 0, 0) : out; // out
  tk_ivec_clear(out);
  if (lua_type(L, 2) == LUA_TNUMBER) {
    uid = tk_lua_checkinteger(L, 2, "id");
    size_t n = 0;
    int64_t *data = tk_inv_get(I, uid, &n);
    if (!n)
      return 1;
    tk_ivec_ensure(out, n);
    memcpy(out->a, data, n * sizeof(int64_t));
    out->n = n;
  } else {
    uids = tk_ivec_peek(L, 2, "uids");
    for (uint64_t i = 0; i < uids->n; i ++) {
      uid = uids->a[i];
      size_t n = 0;
      int64_t *data = tk_inv_get(I, uid, &n);
      if (!n)
        continue;
      tk_ivec_ensure(out, out->n + n);
      memcpy(out->a + out->n, data, n * sizeof(int64_t));
      out->n += n;
    }
  }
  return 1;
}

static inline int tk_inv_neighborhoods_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  uint64_t knn = tk_lua_checkunsigned(L, 2, "knn");
  double eps = tk_lua_optposdouble(L, 3, "eps", 1.0);
  const char *typ = tk_lua_optstring(L, 4, "comparator", "jaccard");
  tk_inv_cmp_type_t cmp = TK_INV_JACCARD;
  if (!strcmp(typ, "jaccard"))
    cmp = TK_INV_JACCARD;
  else if (!strcmp(typ, "overlap"))
    cmp = TK_INV_OVERLAP;
  else
    tk_lua_verror(L, 3, "neighbors", "invalid comparator specified", typ);
  tk_inv_neighborhoods(L, I, knn, eps, cmp, 0, 0);
  return 2;
}

static inline int tk_inv_neighbors_lua (lua_State *L)
{
  lua_settop(L, 6);
  tk_inv_t *I = tk_inv_peek(L, 1);
  uint64_t knn = tk_lua_optunsigned(L, 3, "knn", 0);
  double eps = tk_lua_optposdouble(L, 4, "eps", 1.0);
  tk_rvec_t *out = tk_rvec_peek(L, 5, "out");
  const char *typ = tk_lua_optstring(L, 6, "comparator", "jaccard");
  tk_inv_cmp_type_t cmp = TK_INV_JACCARD;
  if (!strcmp(typ, "jaccard"))
    cmp = TK_INV_JACCARD;
  else if (!strcmp(typ, "overlap"))
    cmp = TK_INV_OVERLAP;
  else
    tk_lua_verror(L, 3, "neighbors", "invalid comparator specified", typ);
  if (lua_type(L, 2) == LUA_TNUMBER) {
    int64_t uid = tk_lua_checkinteger(L, 2, "id");
    tk_inv_neighbors_by_id(I, uid, knn, eps, out, cmp);
  } else {
    tk_ivec_t *vec = tk_ivec_peek(L, 2, "vector");
    tk_inv_neighbors_by_vec(I, vec->a, vec->n, -1, knn, eps, out, cmp);
  }
  return 0;
}

static inline int tk_inv_size_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  lua_pushinteger(L, (int64_t) tk_inv_size(I));
  return 1;
}

static inline int tk_inv_threads_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  lua_pushinteger(L, (int64_t) I->pool->n_threads);
  return 1;
}

static inline int tk_inv_features_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  lua_pushinteger(L, (int64_t) I->features);
  return 1;
}

static inline int tk_inv_persist_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  FILE *fh;
  int t = lua_type(L, 2);
  bool tostr = t == LUA_TBOOLEAN && lua_toboolean(L, 2);
  if (t == LUA_TSTRING)
    fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  else if (tostr)
    fh = tk_lua_tmpfile(L);
  else
    return tk_lua_verror(L, 2, "persist", "expecting either a filepath or true (for string serialization)");
  tk_inv_persist(L, I, fh);
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

static inline int tk_inv_destroy_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  tk_inv_destroy(I);
  return 0;
}

static inline int tk_inv_shrink_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  tk_inv_shrink(I);
  return 0;
}

static inline int tk_inv_ids_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  tk_iumap_keys(L, I->uid_sid);
  return 1;
}

static inline void tk_inv_worker (void *dp, int sig)
{
  tk_inv_stage_t stage = (tk_inv_stage_t) sig;
  tk_inv_thread_t *data = (tk_inv_thread_t *) dp;
  tk_inv_t *I = data->I;
  tk_ivec_t *cnt = data->cnt;
  tk_ivec_t *touched = data->touched;
  tk_inv_hoods_t *hoods = data->hoods;
  tk_ivec_t *sids = data->sids;
  tk_iumap_t *sid_idx = data->sid_idx;
  double eps = data->eps;
  uint64_t knn = data->knn;
  tk_inv_cmp_type_t cmp = data->cmp;
  khint_t khi;
  int64_t usid, vsid, fid, iv;
  int64_t start, end;
  int64_t *ubits, *vbits;
  size_t nubits, nvbits;
  tk_rvec_t *uhood;
  tk_ivec_t *vsids;
  switch (stage) {
    case TK_INV_NEIGHBORHOODS:
      touched->n = 0;
      for (int64_t i = (int64_t) data->ifirst; i <= (int64_t) data->ilast; i ++) {
        usid = sids->a[i];
        if (tk_iumap_get(I->sid_uid, usid) == tk_iumap_end(I->sid_uid))
          continue;
        ubits = tk_inv_sget(I, usid, &nubits);
        uhood = hoods->a[i];
        if (knn) {
          tk_rvec_clear(uhood);
          tk_rvec_ensure(uhood, knn);
          uhood->m = knn;
        }
        start = I->node_offsets->a[usid];
        end = (usid + 1 == (int64_t)I->node_offsets->n)
          ? (int64_t) I->node_bits->n
          : I->node_offsets->a[usid + 1];
        for (int64_t j = start; j < end; j ++) {
          fid = I->node_bits->a[j];
          vsids = I->postings->a[fid];
          for (uint64_t k = 0; k < vsids->n; k ++) {
            vsid = vsids->a[k];
            if (vsid == usid)
              continue;
            khi = tk_iumap_get(sid_idx, vsid);        // only consider slice
            if (khi == tk_iumap_end(sid_idx))
              continue;
            iv = tk_iumap_value(sid_idx, khi);
            if (cnt->a[iv] == 0)
              tk_ivec_push(touched, iv);
            cnt->a[iv] ++;
          }
        }
        for (uint64_t ti = 0; ti < touched->n; ti ++) {
          iv = touched->a[ti];
          vsid = sids->a[iv];
          size_t inter = (size_t) cnt->a[iv];
          vbits = tk_inv_sget(I, vsid, &nvbits);
          double sim  = tk_inv_similarity(inter, nubits, nvbits, cmp);
          double dist = tk_inv_length_bump(1.0 - sim, nvbits);
          if (dist <= eps) {
            int64_t vuid = tk_inv_sid_uid(I, vsid);
            if (vuid >= 0) {
              if (knn)
                tk_rvec_hmax(uhood, tk_rank(vuid, dist));
              else
                tk_rvec_push(uhood, tk_rank(vuid, dist));
            }
          }
        }
        tk_rvec_asc(uhood, 0, uhood->n);
        for (uint64_t i = 0; i < uhood->n; ++i) {
          size_t len = 0;
          tk_inv_sget(I,
            tk_inv_uid_sid(I, uhood->a[i].i, false),
            &len);
          uhood->a[i].d = tk_inv_length_unbump(uhood->a[i].d, len);
        }
        uhood->m = uhood->n;
        uhood->a = realloc(uhood->a, uhood->n * sizeof(*uhood->a));
        for (uint64_t ti = 0; ti < touched->n; ti ++)
          cnt->a[touched->a[ti]] = 0;
        touched->n = 0;
      }
      break;
  }
}

static luaL_Reg tk_inv_lua_mt_fns[] =
{
  { "add", tk_inv_add_lua },
  { "remove", tk_inv_remove_lua },
  { "get", tk_inv_get_lua },
  { "neighborhoods", tk_inv_neighborhoods_lua },
  { "neighbors", tk_inv_neighbors_lua },
  { "size", tk_inv_size_lua },
  { "threads", tk_inv_threads_lua },
  { "features", tk_inv_features_lua },
  { "persist", tk_inv_persist_lua },
  { "destroy", tk_inv_destroy_lua },
  { "shrink", tk_inv_shrink_lua },
  { "ids", tk_inv_ids_lua },
  { NULL, NULL }
};

static inline void tk_inv_suppress_unused_lua_mt_fns (void)
  { (void) tk_inv_lua_mt_fns; }

static inline tk_inv_t *tk_inv_create (
  lua_State *L,
  uint64_t features,
  uint64_t n_threads
) {
  tk_inv_t *I = tk_lua_newuserdata(L, tk_inv_t, TK_INV_MT, tk_inv_lua_mt_fns, tk_inv_gc_lua);
  int Ii = tk_lua_absindex(L, -1);
  I->destroyed = false;
  I->next_sid = 0;
  I->features = features;
  I->uid_sid = tk_iumap_create();
  I->sid_uid = tk_iumap_create();
  I->node_offsets = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, -2, -1);
  lua_pop(L, 1);
  I->node_bits = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, -2, -1);
  lua_pop(L, 1);
  I->postings = tk_inv_postings_create(L, features, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, -2, -1);
  for (uint64_t i = 0; i < features; i ++) {
    I->postings->a[i] = tk_ivec_create(L, 0, 0, 0);
    tk_lua_add_ephemeron(L, TK_INV_EPH, -2, -1);
    lua_pop(L, 1);
  }
  lua_pop(L, 1);
  I->threads = tk_malloc(L, n_threads * sizeof(tk_inv_thread_t));
  memset(I->threads, 0, n_threads * sizeof(tk_inv_thread_t));
  I->pool = tk_threads_create(L, n_threads, tk_inv_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    tk_inv_thread_t *data = I->threads + i;
    I->pool->threads[i].data = data;
    data->I = I;
    data->seen = tk_iuset_create();
    data->cnt = tk_ivec_create(L, 0, 0, 0);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
    data->touched = tk_ivec_create(L, 0, 0, 0);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
  }
  return I;
}

static inline tk_inv_t *tk_inv_load (
  lua_State *L,
  FILE *fh,
  uint64_t n_threads
) {
  // userdata + metatable
  tk_inv_t *I = tk_lua_newuserdata(L, tk_inv_t, TK_INV_MT, tk_inv_lua_mt_fns, tk_inv_gc_lua);
  int Ii = tk_lua_absindex(L, -1);
  memset(I, 0, sizeof(tk_inv_t));
  // core scalars
  tk_lua_fread(L, &I->destroyed, sizeof(bool), 1, fh);
  if (I->destroyed)
    tk_lua_verror(L, 2, "load", "index was destroyed when saved");
  tk_lua_fread(L, &I->next_sid,  sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &I->features,  sizeof(uint64_t), 1, fh);
  // uid → sid map
  I->uid_sid = tk_iumap_create();
  khint_t nkeys, k; int absent;
  tk_lua_fread(L, &nkeys, sizeof(khint_t), 1, fh);
  for (khint_t i = 0; i < nkeys; i ++) {
    int64_t key, val;
    tk_lua_fread(L, &key, sizeof(int64_t), 1, fh);
    tk_lua_fread(L, &val, sizeof(int64_t), 1, fh);
    k = tk_iumap_put(I->uid_sid, key, &absent);
    tk_iumap_value(I->uid_sid, k) = val;
  }
  // sid → uid map
  I->sid_uid = tk_iumap_create();
  tk_lua_fread(L, &nkeys, sizeof(khint_t), 1, fh);
  for (khint_t i = 0; i < nkeys; i ++) {
    int64_t key, val;
    tk_lua_fread(L, &key, sizeof(int64_t), 1, fh);
    tk_lua_fread(L, &val, sizeof(int64_t), 1, fh);
    k = tk_iumap_put(I->sid_uid, key, &absent);
    tk_iumap_value(I->sid_uid, k) = val;
  }
  // node_offsets
  uint64_t n = 0;
  tk_lua_fread(L, &n, sizeof(uint64_t), 1, fh);
  I->node_offsets = tk_ivec_create(L, n, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  if (n) tk_lua_fread(L, I->node_offsets->a, sizeof(int64_t), n, fh);
  lua_pop(L, 1);
  // node_bits
  tk_lua_fread(L, &n, sizeof(uint64_t), 1, fh);
  I->node_bits = tk_ivec_create(L, n, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  if (n) tk_lua_fread(L, I->node_bits->a, sizeof(int64_t), n, fh);
  lua_pop(L, 1);
  // postings vector
  uint64_t pcount = 0;
  tk_lua_fread(L, &pcount, sizeof(uint64_t), 1, fh);
  I->postings = tk_inv_postings_create(L, pcount, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  for (uint64_t i = 0; i < pcount; i ++) {
    uint64_t plen;
    tk_lua_fread(L, &plen, sizeof(uint64_t), 1, fh);
    tk_inv_posting_t P = tk_ivec_create(L, plen, 0, 0);
    if (plen)
      tk_lua_fread(L, P->a, sizeof(int64_t), plen, fh);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
    I->postings->a[i] = P;
  }
  lua_pop(L, 1); // pop postings vector
  // thread pool
  I->threads = tk_malloc(L, n_threads * sizeof(tk_inv_thread_t));
  memset(I->threads, 0, n_threads * sizeof(tk_inv_thread_t));
  I->pool = tk_threads_create(L, n_threads, tk_inv_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    tk_inv_thread_t *th = I->threads + i;
    I->pool->threads[i].data = th;
    th->I = I;
    th->seen = tk_iuset_create();
    th->cnt = tk_ivec_create(L, 0, 0, 0);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
    th->touched = tk_ivec_create(L, 0, 0, 0);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
  }
  return I;
}

#endif
