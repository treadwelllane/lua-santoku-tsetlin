#ifndef TK_GRAPH_H
#define TK_GRAPH_H

#include <santoku/tsetlin/conf.h>
#include <santoku/threads.h>
#include <santoku/ivec.h>
#include <santoku/iumap.h>
#include <santoku/pumap.h>
#include <santoku/tsetlin/inv.h>
#include <santoku/tsetlin/ann.h>
#include <santoku/tsetlin/hbi.h>
#include <santoku/tsetlin/dsu.h>
#include <float.h>

#define TK_GRAPH_MT "tk_graph_t"
#define TK_GRAPH_EPH "tk_graph_eph"

typedef tk_iuset_t * tk_graph_adj_item_t;
#define tk_vec_name tk_graph_adj
#define tk_vec_base tk_graph_adj_item_t
#define tk_vec_destroy_item(...) tk_iuset_destroy(__VA_ARGS__)
#define tk_vec_limited
#include <santoku/vec/tpl.h>

typedef enum {
  TK_GRAPH_TODO
} tk_graph_stage_t;

typedef struct tk_graph_thread_s tk_graph_thread_t;

typedef struct tk_graph_s {
  tm_pairs_t *pairs;
  tk_ivec_t *uids;
  tk_ivec_t *uids_hoods;
  tk_iumap_t *uids_idx;
  tk_iumap_t *uids_hoods_idx;
  tk_graph_adj_t *adj_pos, *adj_neg;
  tk_inv_t *inv;
  tk_ann_t *ann;
  tk_hbi_t *hbi;
  tk_inv_hoods_t *inv_hoods;
  tk_ann_hoods_t *ann_hoods;
  tk_hbi_hoods_t *hbi_hoods;
  uint64_t knn_cache;
  double knn_eps;
  tk_pvec_t *pos, *neg;
  tk_ivec_t *labels;
  uint64_t n_pos, n_neg;
  tk_dsu_t dsu;
  tk_graph_thread_t *threads;
  tk_threadpool_t *pool;
} tk_graph_t;

typedef struct tk_graph_thread_s {
  tk_graph_t *graph;
} tk_graph_thread_t;

static inline tk_graph_t *tk_graph_peek (lua_State *L, int i)
{
  return (tk_graph_t *) luaL_checkudata(L, i, TK_GRAPH_MT);
}

#endif
