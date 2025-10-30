#include <santoku/tsetlin/graph.h>
#include <santoku/cvec.h>

static inline tk_pvec_t *tm_add_anchor_edges_immediate(
  lua_State *L,
  int Gi,
  tk_graph_t *graph
);

static inline tk_graph_t *tm_graph_create (
  lua_State *L,
  tk_pvec_t *edges,
  tk_inv_t *knn_inv,
  tk_ann_t *knn_ann,
  tk_hbi_t *knn_hbi,
  tk_ivec_sim_type_t knn_cmp,
  double knn_cmp_alpha,
  double knn_cmp_beta,
  int64_t knn_rank,
  tk_inv_t *category_inv,
  tk_ivec_sim_type_t category_cmp,
  double category_alpha,
  double category_beta,
  uint64_t category_anchors,
  uint64_t category_knn,
  double category_knn_decay,
  int64_t category_ranks,
  tk_inv_t *weight_inv,
  tk_ann_t *weight_ann,
  tk_hbi_t *weight_hbi,
  tk_ivec_sim_type_t weight_cmp,
  double weight_alpha,
  double weight_beta,
  tk_graph_weight_pooling_t weight_pooling,
  uint64_t random_pairs,
  double weight_eps,
  tk_graph_reweight_t reweight,
  int64_t sigma_k,
  double sigma_scale,
  uint64_t knn,
  uint64_t knn_cache,
  double knn_eps,
  bool bridge,
  uint64_t probe_radius
);

static inline void tk_graph_add_adj (
  tk_graph_t *graph,
  int64_t u,
  int64_t v
) {
  int kha;
  uint32_t khi;
  khi = tk_iumap_get(graph->uids_idx, u);
  if (khi == tk_iumap_end(graph->uids_idx))
    return;
  int64_t iu = tk_iumap_val(graph->uids_idx, khi);
  khi = tk_iumap_get(graph->uids_idx, v);
  if (khi == tk_iumap_end(graph->uids_idx))
    return;
  int64_t iv = tk_iumap_val(graph->uids_idx, khi);
  tk_iuset_put(graph->adj->a[iu], iv, &kha);
  tk_iuset_put(graph->adj->a[iv], iu, &kha);
}

static inline void tm_add_knn (
  lua_State *L,
  tk_graph_t *graph
) {
  int kha;
  uint32_t khi;
  uint64_t knn = graph->knn;
  if (!graph->uids_hoods)
    return;
  uint64_t max_hoods = graph->uids_hoods ? graph->uids_hoods->n : 0;
  for (uint64_t hood_idx = 0; hood_idx < max_hoods; hood_idx++) {
    int64_t u = graph->uids_hoods->a[hood_idx];
    uint32_t u_khi = tk_iumap_get(graph->uids_idx, u);
    if (u_khi == tk_iumap_end(graph->uids_idx))
      continue;

    uint64_t rem = knn;
    int64_t neighbor_idx;
    int64_t v;

    TK_GRAPH_FOREACH_HOOD_NEIGHBOR(graph->knn_inv, graph->knn_ann, graph->knn_hbi,
                                   graph->knn_inv_hoods, graph->knn_ann_hoods,
                                   graph->knn_hbi_hoods, hood_idx,
                                   graph->knn_eps, graph->uids_hoods,
                                   neighbor_idx, v, {
      if (!rem) break;
      uint32_t v_khi = tk_iumap_get(graph->uids_idx, v);
      if (v_khi == tk_iumap_end(graph->uids_idx))
        continue;
      tk_edge_t e = tk_edge(u, v, 0.0);
      khi = tk_euset_put(graph->pairs, e, &kha);
      if (!kha)
        continue;
      tk_graph_add_adj(graph, u, v);
      tk_dsu_union(graph->dsu, u, v);
      graph->n_edges++;
      rem--;
    });
  }
}

static inline tk_evec_t *tm_mst_knn_candidates (
  lua_State *L,
  tk_graph_t *graph
) {
  tk_evec_t *all_candidates = tk_evec_create(0, 0, 0, 0);
  if (!graph->uids_hoods)
    return all_candidates;
  if (graph->knn_inv == NULL && graph->knn_ann == NULL && graph->knn_hbi == NULL)
    return all_candidates;

  uint32_t khi;
  uint64_t max_hoods = graph->uids_hoods->n;

  for (uint64_t hood_idx = 0; hood_idx < max_hoods; hood_idx++) {
    int64_t u = graph->uids_hoods->a[hood_idx];
    int64_t cu = tk_dsu_find(graph->dsu, u);
    int64_t neighbor_idx;
    int64_t v;
    TK_GRAPH_FOREACH_HOOD_NEIGHBOR(graph->knn_inv, graph->knn_ann, graph->knn_hbi,
                                   graph->knn_inv_hoods, graph->knn_ann_hoods,
                                   graph->knn_hbi_hoods, hood_idx,
                                   graph->knn_eps, graph->uids_hoods,
                                   neighbor_idx, v, {
      if (cu == tk_dsu_find(graph->dsu, v))
        continue;
      tk_edge_t e = tk_edge(u, v, 0.0);
      khi = tk_euset_get(graph->pairs, e);
      if (khi != tk_euset_end(graph->pairs))
        continue;
      double d = tk_graph_distance(graph, u, v, graph->q_weights, graph->e_weights, graph->inter_weights);
      if (tk_evec_push(all_candidates, tk_edge(u, v, d)) != 0) {
        tk_evec_destroy(all_candidates);
        return NULL;
      }
    });
  }

  tk_evec_asc(all_candidates, 0, all_candidates->n);
  return all_candidates;
}

static inline void tm_add_mst (
  lua_State *L,
  int Gi,
  tk_graph_t *graph,
  tk_evec_t *candidates
) {
  if (candidates != NULL) {
    int kha;
    uint32_t khi;
    for (uint64_t i = 0; i < candidates->n && tk_dsu_components(graph->dsu) > 1; i ++) {
      tk_edge_t c = candidates->a[i];
      int64_t cu = tk_dsu_find(graph->dsu, c.u);
      int64_t cv = tk_dsu_find(graph->dsu, c.v);
      if (cu == cv)
        continue;
      tk_edge_t e = tk_edge(c.u, c.v, 0.0);
      khi = tk_euset_put(graph->pairs, e, &kha);
      if (!kha)
        continue;
      tk_graph_add_adj(graph, c.u, c.v);
      tk_dsu_union(graph->dsu, c.u, c.v);
      graph->n_edges ++;
    }
  } else if (graph->bridge) {
    tk_pumap_t *reps_comp = tk_pumap_create(L, 0);
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
    lua_pop(L, 1);
    for (int64_t idx = 0; idx < (int64_t) graph->uids->n; idx ++) {
      int64_t u = graph->uids->a[idx];
      int64_t comp = tk_dsu_find(graph->dsu, u);
      uint32_t kc;
      int is_new;
      int64_t deg = tk_iuset_size(graph->adj->a[idx]);
      kc = tk_pumap_put(reps_comp, comp, &is_new);
      if (is_new || deg > tk_pumap_val(reps_comp, kc).p)
        tk_pumap_setval(reps_comp, kc, tk_pair(idx, deg));
    }
    tk_pvec_t *centers = tk_pumap_values(L, reps_comp);
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
    lua_pop(L, 1);
    assert(centers->n > 1);
    tk_pvec_shuffle(centers);
    int kha;
    for (int64_t i = centers->n > 2 ? 0 : 1; i < (int64_t) centers->n; i ++) {
      int64_t j = i == ((int64_t) centers->n - 1) ? 0 : i + 1;
      int64_t iu = centers->a[i].i;
      int64_t iv = centers->a[j].i;
      int64_t u = graph->uids->a[iu];
      int64_t v = graph->uids->a[iv];
      tk_edge_t e = tk_edge(u, v, 0.0);
      tk_euset_put(graph->pairs, e, &kha);
      if (kha) {
        tk_graph_add_adj(graph, u, v);
        tk_dsu_union(graph->dsu, u, v);
        graph->n_edges ++;
      }
    }
  }
}

static inline void tm_init_uids (
  lua_State *L,
  int Gi,
  tk_graph_t *graph
) {
  graph->uids = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);
  graph->uids_idx = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);
}

static inline void tm_process_seed_edges (
  lua_State *L,
  tk_graph_t *graph
) {
  if (graph->edges == NULL || graph->edges->n == 0)
    return;
  int kha;
  for (uint64_t i = 0; i < graph->edges->n; i++) {
    int64_t u = graph->edges->a[i].i;
    int64_t v = graph->edges->a[i].p;
    uint32_t khi = tk_iumap_put(graph->uids_idx, u, &kha);
    if (kha) {
      tk_iumap_setval(graph->uids_idx, khi, (int64_t) graph->uids->n);
      if (tk_ivec_push(graph->uids, u) != 0) {
        tk_lua_verror(L, 2, "process_seed_edges", "allocation failed");
        return;
      }
    }
    khi = tk_iumap_put(graph->uids_idx, v, &kha);
    if (kha) {
      tk_iumap_setval(graph->uids_idx, khi, (int64_t) graph->uids->n);
      if (tk_ivec_push(graph->uids, v) != 0) {
        tk_lua_verror(L, 2, "process_seed_edges", "allocation failed");
        return;
      }
    }
  }

}

static inline void tm_add_seed_edges_immediate(
  lua_State *L,
  tk_graph_t *graph
) {
  if (graph->edges == NULL || graph->edges->n == 0)
    return;
  int kha;
  uint32_t khi;
  for (uint64_t i = 0; i < graph->edges->n; i++) {
    int64_t u = graph->edges->a[i].i;
    int64_t v = graph->edges->a[i].p;
    uint32_t khi_u = tk_iumap_get(graph->uids_idx, u);
    uint32_t khi_v = tk_iumap_get(graph->uids_idx, v);
    if (khi_u == tk_iumap_end(graph->uids_idx) ||
        khi_v == tk_iumap_end(graph->uids_idx)) {
      continue;
    }
    tk_edge_t e = tk_edge(u, v, 0.0);
    khi = tk_euset_put(graph->pairs, e, &kha);
    if (!kha)
      continue;
    tk_graph_add_adj(graph, u, v);
    tk_dsu_union(graph->dsu, u, v);
    graph->n_edges++;
  }
}

static inline tk_pvec_t *tm_add_anchor_edges_immediate(
  lua_State *L,
  int Gi,
  tk_graph_t *graph
) {
  if (graph->category_inv == NULL || graph->category_anchors == 0)
    return NULL;

  tk_pvec_t *new_edges = tk_pvec_create(L, 0, 0, 0);
  if (!new_edges) {
    tk_lua_verror(L, 2, "add_anchor_edges", "allocation failed");
    return NULL;
  }

  tk_inv_t *inv = graph->category_inv;
  uint64_t category_anchors = graph->category_anchors;
  double category_knn_decay = graph->category_knn_decay;
  double category_knn = graph->category_knn;
  double category_cmp = graph->category_cmp;
  double category_alpha = graph->category_alpha;
  double category_beta = graph->category_beta;
  int64_t max_rank = (int64_t)inv->n_ranks - 1;

  tk_ivec_t *rank_features = NULL;
  if (inv->ranks != NULL && graph->category_ranks > 0 && graph->category_ranks < (int64_t)inv->n_ranks) {
    rank_features = tk_ivec_create(L, 0, 0, 0);
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
    lua_pop(L, 1);
    for (uint64_t i = 0; i < inv->ranks->n; i++)
      if (inv->ranks->a[i] < graph->category_ranks)
        tk_ivec_push(rank_features, (int64_t)i);
  }

  uint64_t n_features = rank_features != NULL ? rank_features->n : inv->features;
  bool need_buffers = inv && inv->ranks && inv->n_ranks > 1;
  bool has_error = false;

  #pragma omp parallel reduction(||:has_error)
  {
    tk_pvec_t *local_pairs = tk_pvec_create(NULL, 0, 0, 0);
    tk_rvec_t *knn_heap = tk_rvec_create(NULL, 0, 0, 0);
    tk_ivec_t *anchors = tk_ivec_create(NULL, 0, 0, 0);
    tk_dvec_t *q_weights = NULL;
    tk_dvec_t *e_weights = NULL;
    tk_dvec_t *inter_weights = NULL;

    if (!local_pairs || !knn_heap || !anchors) {
      has_error = true;
    } else {
      if (need_buffers) {
        q_weights = tk_dvec_create(NULL, 0, 0, 0);
        e_weights = tk_dvec_create(NULL, 0, 0, 0);
        inter_weights = tk_dvec_create(NULL, 0, 0, 0);
        if (!q_weights || !e_weights || !inter_weights) {
          has_error = true;
        }
      }

      if (!has_error) {
        #pragma omp for schedule(static)
        for (uint64_t fi = 0; fi < n_features; fi++) {
          if (has_error) continue;

          int64_t f = rank_features == NULL ? (int64_t)fi : rank_features->a[fi];
          tk_ivec_t *postings = inv->postings->a[f];
          if (!postings || postings->n == 0)
            continue;

          int64_t rank = inv->ranks ? inv->ranks->a[f] : 0;
          int64_t effective_rank = (category_knn_decay < 0.0) ? (max_rank - rank) : rank;
          double knn_weight = exp(-(double)effective_rank * fabs(category_knn_decay));
          uint64_t n_anchors = (uint64_t)category_anchors;
          uint64_t n_wanted = category_knn > 0 ? category_knn : n_anchors;
          uint64_t k_nearest = (uint64_t)ceil((double)n_wanted * knn_weight);

          tk_ivec_clear(anchors);
          for (uint64_t i = 0; i < postings->n; i++) {
            int64_t sid = postings->a[i];
            if (sid >= 0 && sid < (int64_t)inv->sid_to_uid->n) {
              int64_t uid = inv->sid_to_uid->a[sid];
              if (uid >= 0) {
                if (tk_ivec_push(anchors, uid) != 0) {
                  has_error = true;
                  break;
                }
              }
            }
          }
          if (has_error) continue;
          if (anchors->n == 0)
            continue;

          tk_ivec_shuffle(anchors);
          for (uint64_t i = 0; i < anchors->n; i++) {
            int64_t u = anchors->a[i];
            tk_rvec_clear(knn_heap);
            for (uint64_t j = 0; j < anchors->n && j < n_anchors; j++) {
              if (i == j) continue;
              int64_t v = anchors->a[j];
              double dist = tk_inv_distance(inv, u, v, category_cmp, category_alpha, category_beta, q_weights, e_weights, inter_weights);
              tk_rvec_hmin(knn_heap, k_nearest, tk_rank(v, dist));
            }
            for (uint64_t k = 0; k < knn_heap->n; k++) {
              int64_t v = knn_heap->a[k].i;
              if (tk_pvec_push(local_pairs, tk_pair(u, v)) != 0) {
                has_error = true;
                break;
              }
            }
            if (has_error) break;
          }
        }

        #pragma omp critical
        {
          for (uint64_t j = 0; j < local_pairs->n; j++) {
            int64_t u = local_pairs->a[j].i;
            int64_t v = local_pairs->a[j].p;
            uint32_t khi_u = tk_iumap_get(graph->uids_idx, u);
            uint32_t khi_v = tk_iumap_get(graph->uids_idx, v);
            int kha;
            uint32_t khi;

            if (khi_u == tk_iumap_end(graph->uids_idx)) {
              khi_u = tk_iumap_put(graph->uids_idx, u, &kha);
              if (kha) {
                tk_iumap_setval(graph->uids_idx, khi_u, (int64_t)graph->uids->n);
                if (tk_ivec_push(graph->uids, u) != 0) {
                  has_error = true;
                }
              }
            }
            if (!has_error && khi_v == tk_iumap_end(graph->uids_idx)) {
              khi_v = tk_iumap_put(graph->uids_idx, v, &kha);
              if (kha) {
                tk_iumap_setval(graph->uids_idx, khi_v, (int64_t)graph->uids->n);
                if (tk_ivec_push(graph->uids, v) != 0) {
                  has_error = true;
                }
              }
            }
            if (!has_error) {
              tk_edge_t e = tk_edge(u, v, 0.0);
              khi = tk_euset_put(graph->pairs, e, &kha);
              if (kha) {
                if (tk_pvec_push(new_edges, tk_pair(u, v)) != 0) {
                  has_error = true;
                } else {
                  graph->n_edges++;
                }
              }
            }
            if (has_error) break;
          }
        }
      }
    }

    if (local_pairs) tk_pvec_destroy(local_pairs);
    if (knn_heap) tk_rvec_destroy(knn_heap);
    if (anchors) tk_ivec_destroy(anchors);
    if (q_weights) tk_dvec_destroy(q_weights);
    if (e_weights) tk_dvec_destroy(e_weights);
    if (inter_weights) tk_dvec_destroy(inter_weights);
  }

  if (has_error) {
    tk_lua_verror(L, 2, "add_anchor_edges", "worker allocation failed");
  }

  return new_edges;
}

static inline void tm_adj_init (
  lua_State *L,
  int Gi,
  tk_graph_t *graph
) {
  graph->adj = tk_graph_adj_create(L, graph->uids->n, 0, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);
  for (uint64_t i = 0; i < graph->uids->n; i ++) {
    graph->adj->a[i] = tk_iuset_create(L, 0);
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
    lua_pop(L, 1);
  }
}

static inline void tm_adj_resize (
  lua_State *L,
  int Gi,
  tk_graph_t *graph
) {
  uint64_t old_size = graph->adj->n;
  uint64_t new_size = graph->uids->n;
  if (new_size <= old_size)
    return;
  tk_graph_adj_resize(graph->adj, new_size, true);
  for (uint64_t i = old_size; i < new_size; i++) {
    graph->adj->a[i] = tk_iuset_create(L, 0);
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
    lua_pop(L, 1);
  }
}

static inline void tm_run_knn_queries (
  lua_State *L,
  int Gi,
  tk_graph_t *graph
) {
  bool have_index = graph->knn_inv != NULL || graph->knn_ann != NULL || graph->knn_hbi != NULL;
  if (!graph->knn || !graph->knn_cache || !have_index)
    return;
  TK_INDEX_NEIGHBORHOODS(L,
    graph->knn_inv, graph->knn_ann, graph->knn_hbi,
    graph->knn_cache, graph->probe_radius, 1.0,
    graph->knn_cmp, graph->knn_cmp_alpha, graph->knn_cmp_beta, graph->knn_rank,
    &graph->knn_inv_hoods, &graph->knn_ann_hoods, &graph->knn_hbi_hoods, &graph->uids_hoods);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -2);
  lua_pop(L, 2);
  if (graph->uids_hoods) {
    graph->uids_idx_hoods = tk_iumap_from_ivec(L, graph->uids_hoods);
    if (!graph->uids_idx_hoods)
      tk_error(L, "graph_create: iumap_from_ivec failed", ENOMEM);
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
    lua_pop(L, 1);
    int kha;
    for (uint64_t i = 0; i < graph->uids_hoods->n; i++) {
      int64_t uid = graph->uids_hoods->a[i];
      uint32_t khi = tk_iumap_put(graph->uids_idx, uid, &kha);
      if (kha) {
        tk_iumap_setval(graph->uids_idx, khi, (int64_t) graph->uids->n);
        if (tk_ivec_push(graph->uids, uid) != 0) {
          tk_lua_verror(L, 2, "graph_adj", "allocation failed");
          return;
        }
      }
    }
  }
}

static inline void tm_compute_base_distances (
  lua_State *L,
  int Gi,
  tk_graph_t *graph
) {
  if (!graph->weight_inv && !graph->weight_ann && !graph->weight_hbi &&
    !graph->category_inv && !graph->knn_inv && !graph->knn_ann && !graph->knn_hbi)
    return;
  bool need_buffers = (graph->weight_inv && graph->weight_inv->ranks && graph->weight_inv->n_ranks > 1) ||
    (graph->category_inv && graph->category_inv->ranks && graph->category_inv->n_ranks > 1) ||
    (graph->knn_inv && graph->knn_inv->ranks && graph->knn_inv->n_ranks > 1);
  tk_iumap_t **all_rank_maps = NULL;
  bool has_error = false;
  if (graph->reweight == TK_GRAPH_REWEIGHT_RANK) {
    all_rank_maps = tk_malloc(L, (graph->uids->n + 1) * sizeof(tk_iumap_t *));
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
    lua_pop(L, 1);
    for (uint64_t i = 0; i < graph->uids->n; i++)
      all_rank_maps[i] = NULL;
    #pragma omp parallel reduction(||:has_error)
    {
      tk_dvec_t *q_weights = NULL;
      tk_dvec_t *e_weights = NULL;
      tk_dvec_t *inter_weights = NULL;
      tk_rvec_t *neighbor_distances = tk_rvec_create(NULL, 0, 0, 0);
      tk_iumap_t *rank_map = NULL;
      if (need_buffers) {
        q_weights = tk_dvec_create(NULL, 0, 0, 0);
        e_weights = tk_dvec_create(NULL, 0, 0, 0);
        inter_weights = tk_dvec_create(NULL, 0, 0, 0);
        if (!q_weights || !e_weights || !inter_weights) has_error = true;
      }
      if (!neighbor_distances) has_error = true;
      if (!has_error) {
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < (int64_t)graph->uids->n; i++) {
          if (has_error) continue;
          tk_rvec_clear(neighbor_distances);
          int64_t u = graph->uids->a[i];
          int64_t neighbor_idx;
          tk_umap_foreach_keys(graph->adj->a[i], neighbor_idx, ({
            int64_t v = graph->uids->a[neighbor_idx];
            double d = tk_graph_distance(graph, u, v, q_weights, e_weights, inter_weights);
            if (tk_rvec_push(neighbor_distances, tk_rank(neighbor_idx, d)) != 0) {
              has_error = true;
            }
          }))
          if (has_error) continue;
          tk_rvec_asc(neighbor_distances, 0, neighbor_distances->n);
          rank_map = tk_iumap_create(NULL, neighbor_distances->n);
          if (!rank_map) {
            has_error = true;
            continue;
          }
          int kha;
          uint32_t khi;
          for (uint64_t j = 0; j < neighbor_distances->n; j++) {
            khi = tk_iumap_put(rank_map, neighbor_distances->a[j].i, &kha);
            tk_iumap_setval(rank_map, khi, (int64_t)j);
          }
          all_rank_maps[i] = rank_map;
        }
      }
      if (neighbor_distances) tk_rvec_destroy(neighbor_distances);
      if (need_buffers) {
        if (q_weights) tk_dvec_destroy(q_weights);
        if (e_weights) tk_dvec_destroy(e_weights);
        if (inter_weights) tk_dvec_destroy(inter_weights);
      }
    }
    if (has_error) {
      for(uint64_t i = 0; i < graph->uids->n; i++)
        if (all_rank_maps[i]) tk_iumap_destroy(all_rank_maps[i]);
      tk_lua_verror(L, 2, "compute_base_distances", "worker allocation failed in rank pass");
      return;
    }
  }
  #pragma omp parallel reduction(||:has_error)
  {
    tk_dvec_t *q_weights = NULL;
    tk_dvec_t *e_weights = NULL;
    tk_dvec_t *inter_weights = NULL;
    if (need_buffers && graph->reweight != TK_GRAPH_REWEIGHT_RANK) {
      q_weights = tk_dvec_create(NULL, 0, 0, 0);
      e_weights = tk_dvec_create(NULL, 0, 0, 0);
      inter_weights = tk_dvec_create(NULL, 0, 0, 0);
      if (!q_weights || !e_weights || !inter_weights) has_error = true;
    }
    if (!has_error) {
      #pragma omp for schedule(static)
      for (int64_t i = 0; i < (int64_t)graph->uids->n; i++) {
        if (has_error) continue;
        int64_t u = graph->uids->a[i];
        int64_t neighbor_idx;
        tk_umap_foreach_keys(graph->adj->a[i], neighbor_idx, ({
          int64_t v = graph->uids->a[neighbor_idx];
          if (u < v) {
            tk_edge_t edge_key = tk_edge(u, v, 0);
            uint32_t k = tk_euset_get(graph->pairs, edge_key);
            if (k != tk_euset_end(graph->pairs)) {
              double base_uv, base_vu;
              if (graph->reweight == TK_GRAPH_REWEIGHT_RANK) {
                tk_iumap_t *rank_map_u = all_rank_maps[i];
                uint32_t khi_u = tk_iumap_get(rank_map_u, neighbor_idx);
                int64_t rank_uv = (khi_u == tk_iumap_end(rank_map_u)) ? -1 : tk_iumap_val(rank_map_u, khi_u);
                base_uv = (rank_uv == -1) ? 1.0 : (1.0 - (1.0 / (1.0 + (double)rank_uv)));
                tk_iumap_t *rank_map_v = all_rank_maps[neighbor_idx];
                uint32_t khi_v = tk_iumap_get(rank_map_v, i);
                int64_t rank_vu = (khi_v == tk_iumap_end(rank_map_v)) ? -1 : tk_iumap_val(rank_map_v, khi_v);
                base_vu = (rank_vu == -1) ? 1.0 : (1.0 - (1.0 / (1.0 + (double)rank_vu)));
              } else {
                base_uv = tk_graph_distance(graph, u, v, q_weights, e_weights, inter_weights);
                base_vu = base_uv;
              }
              double base_final;
              if (graph->weight_pooling == TK_GRAPH_WEIGHT_POOL_MIN) {
                base_final = fmax(base_uv, base_vu);
              } else {
                base_final = fmin(base_uv, base_vu);
              }
              kh_key(graph->pairs, k).w = base_final;
            }
          }
        }))
      }
    }
    if (need_buffers && graph->reweight != TK_GRAPH_REWEIGHT_RANK) {
      if (q_weights) tk_dvec_destroy(q_weights);
      if (e_weights) tk_dvec_destroy(e_weights);
      if (inter_weights) tk_dvec_destroy(inter_weights);
    }
  }
  if (all_rank_maps) {
    for(uint64_t i = 0; i < graph->uids->n; i++)
      if (all_rank_maps[i]) tk_iumap_destroy(all_rank_maps[i]);
  }
  if (has_error) {
    tk_lua_verror(L, 2, "compute_base_distances", "worker allocation failed in pooling pass");
  }
}

static inline void tm_compute_sigma (
  lua_State *L,
  int Gi,
  tk_graph_t *graph
) {
  if (!graph->sigma_k || !graph->sigma_scale || graph->sigma_scale <= 0.0)
    return;
  graph->sigmas = tk_dvec_create(L, graph->uids->n, 0, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);
  for (uint64_t i = 0; i < graph->uids->n; i++)
    graph->sigmas->a[i] = 1.0;
  bool need_buffers = (graph->weight_inv && graph->weight_inv->ranks && graph->weight_inv->n_ranks > 1) ||
    (graph->category_inv && graph->category_inv->ranks && graph->category_inv->n_ranks > 1) ||
    (graph->knn_inv && graph->knn_inv->ranks && graph->knn_inv->n_ranks > 1);
  bool has_error = false;
  #pragma omp parallel reduction(||:has_error)
  {
    tk_dvec_t *q_weights = NULL;
    tk_dvec_t *e_weights = NULL;
    tk_dvec_t *inter_weights = NULL;
    tk_rvec_t *neighbor_distances = tk_rvec_create(NULL, 0, 0, 0);
    tk_iuset_t *seen = tk_iuset_create(NULL, 0);
    tk_dvec_t *final_distances = tk_dvec_create(NULL, 0, 0, 0);
    if (!neighbor_distances || !seen || !final_distances) {
      has_error = true;
    } else {
      if (need_buffers) {
        q_weights = tk_dvec_create(NULL, 0, 0, 0);
        e_weights = tk_dvec_create(NULL, 0, 0, 0);
        inter_weights = tk_dvec_create(NULL, 0, 0, 0);
        if (!q_weights || !e_weights || !inter_weights) {
          has_error = true;
        }
      }
      if (!has_error) {
        #pragma omp for schedule(static)
        for (uint64_t i = 0; i < graph->uids->n; i++) {
          if (has_error) continue;
          tk_rvec_clear(neighbor_distances);
          tk_dvec_clear(final_distances);
          tk_iuset_clear(seen);
          int64_t uid = graph->uids->a[i];
          int64_t neighbor_idx;
          tk_umap_foreach_keys(graph->adj->a[i], neighbor_idx, ({
            int64_t neighbor_uid = graph->uids->a[neighbor_idx];
            double d = tk_graph_distance(graph, uid, neighbor_uid, q_weights, e_weights, inter_weights);
            if (d != DBL_MAX) {
              if (tk_rvec_push(neighbor_distances, tk_rank(neighbor_idx, d)) != 0) {
                has_error = true;
              } else {
                int kha;
                tk_iuset_put(seen, neighbor_idx, &kha);
              }
            }
          }))
          if (has_error) continue;
          if (graph->uids_idx_hoods) {
            uint32_t khi = tk_iumap_get(graph->uids_idx_hoods, uid);
            if (khi != tk_iumap_end(graph->uids_idx_hoods)) {
              int64_t hood_idx = tk_iumap_val(graph->uids_idx_hoods, khi);
              uint64_t features_ann = graph->knn_ann ? graph->knn_ann->features : 0;
              uint64_t features_hbi = graph->knn_hbi ? graph->knn_hbi->features : 0;
              if (graph->knn_inv_hoods && (hood_idx) < (int64_t)(graph->knn_inv_hoods)->n) {
                tk_rvec_t *__hood = (graph->knn_inv_hoods)->a[hood_idx];
                for (uint64_t __j = 0; __j < __hood->n; __j++) {
                  if (has_error) break;
                  int64_t __nh_idx = __hood->a[__j].i;
                  if (__nh_idx >= 0 && __nh_idx < (int64_t)(graph->uids_hoods)->n) {
                    int64_t __nh_uid = (graph->uids_hoods)->a[__nh_idx];
                    uint32_t __n_khi = tk_iumap_get((graph->uids_idx), __nh_uid);
                    if (__n_khi != tk_iumap_end((graph->uids_idx))) {
                      int64_t __nh_global_idx = tk_iumap_val((graph->uids_idx), __n_khi);
                      int __kha;
                      tk_iuset_put((seen), __nh_global_idx, &__kha);
                      if (__kha) {
                        double __d = tk_graph_distance((graph), (uid), __nh_uid, (q_weights), (e_weights), (inter_weights));
                        if (__d == DBL_MAX) __d = __hood->a[__j].d;
                        if (tk_rvec_push((neighbor_distances), tk_rank(__nh_global_idx, __d)) != 0) { has_error = true; }
                      }
                    }
                  }
                }
              } else if (graph->knn_ann_hoods && (hood_idx) < (int64_t)(graph->knn_ann_hoods)->n) {
                tk_pvec_t *__hood = (graph->knn_ann_hoods)->a[hood_idx];
                double __denom = (features_ann) ? (double)(features_ann) : 1.0;
                for (uint64_t __j = 0; __j < __hood->n; __j++) {
                  if (has_error) break;
                  int64_t __nh_idx = __hood->a[__j].i;
                  if (__nh_idx >= 0 && __nh_idx < (int64_t)(graph->uids_hoods)->n) {
                    int64_t __nh_uid = (graph->uids_hoods)->a[__nh_idx];
                    uint32_t __n_khi = tk_iumap_get((graph->uids_idx), __nh_uid);
                    if (__n_khi != tk_iumap_end((graph->uids_idx))) {
                      int64_t __nh_global_idx = tk_iumap_val((graph->uids_idx), __n_khi);
                      int __kha;
                      tk_iuset_put((seen), __nh_global_idx, &__kha);
                      if (__kha) {
                        double __d = tk_graph_distance((graph), (uid), __nh_uid, (q_weights), (e_weights), (inter_weights));
                        if (__d == DBL_MAX) __d = (double)__hood->a[__j].p / __denom;
                        if (tk_rvec_push((neighbor_distances), tk_rank(__nh_global_idx, __d)) != 0) { has_error = true; }
                      }
                    }
                  }
                }
              } else if (graph->knn_hbi_hoods && (hood_idx) < (int64_t)(graph->knn_hbi_hoods)->n) {
                tk_pvec_t *__hood = (graph->knn_hbi_hoods)->a[hood_idx];
                double __denom = (features_hbi) ? (double)(features_hbi) : 1.0;
                for (uint64_t __j = 0; __j < __hood->n; __j++) {
                  if (has_error) break;
                  int64_t __nh_idx = __hood->a[__j].i;
                  if (__nh_idx >= 0 && __nh_idx < (int64_t)(graph->uids_hoods)->n) {
                    int64_t __nh_uid = (graph->uids_hoods)->a[__nh_idx];
                    uint32_t __n_khi = tk_iumap_get((graph->uids_idx), __nh_uid);
                    if (__n_khi != tk_iumap_end((graph->uids_idx))) {
                      int64_t __nh_global_idx = tk_iumap_val((graph->uids_idx), __n_khi);
                      int __kha;
                      tk_iuset_put((seen), __nh_global_idx, &__kha);
                      if (__kha) {
                        double __d = tk_graph_distance((graph), (uid), __nh_uid, (q_weights), (e_weights), (inter_weights));
                        if (__d == DBL_MAX) __d = (double)__hood->a[__j].p / __denom;
                        if (tk_rvec_push((neighbor_distances), tk_rank(__nh_global_idx, __d)) != 0) { has_error = true; }
                      }
                    }
                  }
                }
              }
            }
          }
          if (has_error) continue;
          tk_rvec_asc(neighbor_distances, 0, neighbor_distances->n);
          for(uint64_t j = 0; j < neighbor_distances->n; j++) {
            double d;
            if (graph->reweight == TK_GRAPH_REWEIGHT_RANK) {
              d = 1.0 - (1.0 / (1.0 + (double)j));
            } else {
              d = neighbor_distances->a[j].d;
            }
            if (tk_dvec_push(final_distances, d) != 0) {
              has_error = true;
              break;
            }
          }
          if (has_error) continue;
          double sigma = 1.0;
          if (final_distances->n > 0) {
            tk_dvec_asc(final_distances, 0, final_distances->n);
            uint64_t k = (graph->sigma_k > 0) ? (uint64_t)graph->sigma_k : final_distances->n;
            if (k > final_distances->n)
              k = final_distances->n;
            sigma = final_distances->a[k - 1];
          }
          graph->sigmas->a[i] = sigma * graph->sigma_scale;
        }
      }
    }
    if (neighbor_distances) tk_rvec_destroy(neighbor_distances);
    if (seen) tk_iuset_destroy(seen);
    if (final_distances) tk_dvec_destroy(final_distances);
    if (need_buffers) {
      if (q_weights) tk_dvec_destroy(q_weights);
      if (e_weights) tk_dvec_destroy(e_weights);
      if (inter_weights) tk_dvec_destroy(inter_weights);
    }
  }
  if (has_error) {
    tk_lua_verror(L, 2, "compute_sigma", "worker allocation failed");
  }
}

static inline void tm_reweight_all_edges (
  lua_State *L,
  tk_graph_t *graph
) {
  const double eps = graph->weight_eps;
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < (int64_t)graph->uids->n; i++) {
    int64_t u = graph->uids->a[i];
    int64_t neighbor_idx;
    tk_umap_foreach_keys(graph->adj->a[i], neighbor_idx, ({
      int64_t v = graph->uids->a[neighbor_idx];
      if (u < v) {
        tk_edge_t edge_key = tk_edge(u, v, 0);
        uint32_t k = tk_euset_get(graph->pairs, edge_key);
        if (k != tk_euset_end(graph->pairs)) {
          double d = kh_key(graph->pairs, k).w;
          if (d < 0.0) d = 0.0;
          if (d > 1.0) d = 1.0;
          double sim;
          if (graph->sigmas && graph->sigmas->n > 0) {
            double si = (i >= 0 && (uint64_t)i < graph->sigmas->n) ? graph->sigmas->a[i] : eps;
            double sj = (neighbor_idx >= 0 && (uint64_t)neighbor_idx < graph->sigmas->n) ? graph->sigmas->a[neighbor_idx] : eps;
            if (si <= 0.0) si = eps;
            if (sj <= 0.0) sj = eps;
            double s = sqrt(si * sj);
            if (s > 0.0) {
              double s2 = s * s;
              sim = exp(-0.5 * (d * d) / s2);
            } else {
              sim = 1.0 - d;
            }
          } else {
            sim = 1.0 - d;
          }
          if (sim < eps) sim = eps;
          if (sim > 1.0) sim = 1.0;
          kh_key(graph->pairs, k).w = sim;
        }
      }
    }))
  }
}

static void tm_graph_destroy_internal (tk_graph_t *graph)
{
  if (graph->destroyed)
    return;
  graph->destroyed = true;
  tk_dsu_destroy(graph->dsu);
}

static inline int tm_graph_gc (lua_State *L)
{
  tk_graph_t *graph = tk_graph_peek(L, 1);
  tm_graph_destroy_internal(graph);
  return 0;
}

static inline int tm_adjacency (lua_State *L)
{
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

  lua_getfield(L, 1, "edges");
  tk_pvec_t *edges = tk_pvec_peekopt(L, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "knn_index");
  tk_inv_t *knn_inv = tk_inv_peekopt(L, -1);
  tk_ann_t *knn_ann = tk_ann_peekopt(L, -1);
  tk_hbi_t *knn_hbi = tk_hbi_peekopt(L, -1);
  lua_pop(L, 1);

  const char *knn_cmp_str = tk_lua_foptstring(L, 1, "graph", "knn_cmp", "jaccard");
  double knn_cmp_alpha = tk_lua_foptnumber(L, 1, "graph", "knn_cmp_alpha", 0.5);
  double knn_cmp_beta = tk_lua_foptnumber(L, 1, "graph", "knn_cmp_beta", 0.5);
  tk_ivec_sim_type_t knn_cmp = TK_IVEC_JACCARD;
  if (!strcmp(knn_cmp_str, "jaccard"))
    knn_cmp = TK_IVEC_JACCARD;
  else if (!strcmp(knn_cmp_str, "overlap"))
    knn_cmp = TK_IVEC_OVERLAP;
  else if (!strcmp(knn_cmp_str, "tversky"))
    knn_cmp = TK_IVEC_TVERSKY;
  else if (!strcmp(knn_cmp_str, "dice"))
    knn_cmp = TK_IVEC_DICE;
  else
    tk_lua_verror(L, 3, "graph", "invalid knn comparator specified", knn_cmp_str);

  lua_getfield(L, 1, "category_index");
  tk_inv_t *category_inv = tk_inv_peekopt(L, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "weight_index");
  tk_inv_t *weight_inv = tk_inv_peekopt(L, -1);
  tk_ann_t *weight_ann = tk_ann_peekopt(L, -1);
  tk_hbi_t *weight_hbi = tk_hbi_peekopt(L, -1);
  lua_pop(L, 1);

  const char *category_cmp_str = tk_lua_foptstring(L, 1, "graph", "category_cmp", "jaccard");
  double category_alpha = tk_lua_foptnumber(L, 1, "graph", "category_alpha", 0.5);
  double category_beta = tk_lua_foptnumber(L, 1, "graph", "category_beta", 0.5);
  tk_ivec_sim_type_t category_cmp = TK_IVEC_JACCARD;
  if (!strcmp(category_cmp_str, "jaccard"))
    category_cmp = TK_IVEC_JACCARD;
  else if (!strcmp(category_cmp_str, "overlap"))
    category_cmp = TK_IVEC_OVERLAP;
  else if (!strcmp(category_cmp_str, "tversky"))
    category_cmp = TK_IVEC_TVERSKY;
  else if (!strcmp(category_cmp_str, "dice"))
    category_cmp = TK_IVEC_DICE;
  else
    tk_lua_verror(L, 3, "graph", "invalid category comparator specified", category_cmp_str);

  uint64_t category_anchors = tk_lua_foptunsigned(L, 1, "graph", "category_anchors", 0);
  uint64_t category_knn = tk_lua_foptunsigned(L, 1, "graph", "category_knn", 0);
  double category_knn_decay = tk_lua_foptnumber(L, 1, "graph", "category_knn_decay", 0.0);

  const char *weight_cmp_str = tk_lua_foptstring(L, 1, "graph", "weight_cmp", "jaccard");
  double weight_alpha = tk_lua_foptnumber(L, 1, "graph", "weight_alpha", 0.5);
  double weight_beta = tk_lua_foptnumber(L, 1, "graph", "weight_beta", 0.5);
  tk_ivec_sim_type_t weight_cmp = TK_IVEC_JACCARD;
  if (!strcmp(weight_cmp_str, "jaccard"))
    weight_cmp = TK_IVEC_JACCARD;
  else if (!strcmp(weight_cmp_str, "overlap"))
    weight_cmp = TK_IVEC_OVERLAP;
  else if (!strcmp(weight_cmp_str, "tversky"))
    weight_cmp = TK_IVEC_TVERSKY;
  else if (!strcmp(weight_cmp_str, "dice"))
    weight_cmp = TK_IVEC_DICE;
  else
    tk_lua_verror(L, 3, "graph", "invalid weight comparator specified", weight_cmp_str);

  const char *weight_pooling_str = tk_lua_foptstring(L, 1, "graph", "weight_pooling", "min");
  tk_graph_weight_pooling_t weight_pooling = TK_GRAPH_WEIGHT_POOL_MIN;
  if (!strcmp(weight_pooling_str, "min"))
    weight_pooling = TK_GRAPH_WEIGHT_POOL_MIN;
  else if (!strcmp(weight_pooling_str, "max"))
    weight_pooling = TK_GRAPH_WEIGHT_POOL_MAX;
  else
    tk_lua_verror(L, 3, "graph", "invalid weight_pooling specified", weight_pooling_str);

  const char *reweight_str = tk_lua_foptstring(L, 1, "graph", "reweight", NULL);
  tk_graph_reweight_t reweight = TK_GRAPH_REWEIGHT_NONE;
  if (reweight_str != NULL) {
    if (!strcmp(reweight_str, "rank"))
      reweight = TK_GRAPH_REWEIGHT_RANK;
    else
      tk_lua_verror(L, 3, "graph", "invalid reweight specified", reweight_str);
  }

  uint64_t random_pairs = tk_lua_foptunsigned(L, 1, "graph", "random_pairs", 0);

  double weight_eps = tk_lua_foptnumber(L, 1, "graph", "weight_eps", 1e-8);
  int64_t sigma_k = tk_lua_foptinteger(L, 1, "graph", "sigma_k", 0);
  double sigma_scale = tk_lua_foptnumber(L, 1, "graph", "sigma_scale", 1.0);

  uint64_t knn = tk_lua_foptunsigned(L, 1, "graph", "knn", 0);
  uint64_t knn_cache = tk_lua_foptunsigned(L, 1, "graph", "knn_cache", 0);
  double knn_eps = tk_lua_foptposdouble(L, 1, "graph", "knn_eps", 1.0);
  int64_t category_ranks = tk_lua_foptinteger(L, 1, "graph", "category_ranks", -1);

  int64_t knn_rank_explicit = tk_lua_foptinteger(L, 1, "graph", "knn_rank", -2);
  int64_t knn_rank;
  if (knn_rank_explicit == -2) {
    knn_rank = (knn_inv == category_inv) ? category_ranks : -1;
  } else {
    knn_rank = knn_rank_explicit;
  }

  bool bridge = tk_lua_foptboolean(L, 1, "graph", "bridge", false);
  uint64_t probe_radius = tk_lua_foptunsigned(L, 1, "graph", "probe_radius", 3);
  if (knn > knn_cache)
    knn_cache = knn;

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tk_graph_t *graph = tm_graph_create(
    L, edges,
    knn_inv, knn_ann, knn_hbi, knn_cmp, knn_cmp_alpha, knn_cmp_beta, knn_rank,
    category_inv, category_cmp, category_alpha, category_beta,
    category_anchors, category_knn, category_knn_decay, category_ranks,
    weight_inv, weight_ann, weight_hbi, weight_cmp, weight_alpha, weight_beta, weight_pooling,
    random_pairs, weight_eps, reweight, sigma_k, sigma_scale,
    knn, knn_cache, knn_eps, bridge, probe_radius);
  int Gi = tk_lua_absindex(L, -1);

  tm_init_uids(L, Gi, graph);
  tm_process_seed_edges(L, graph);
  graph->dsu = tk_dsu_create(L, graph->uids);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);

  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, (int64_t)graph->uids->n);
    lua_pushinteger(L, tk_dsu_components(graph->dsu));
    lua_pushinteger(L, 0);
    lua_pushstring(L, "init");
    lua_call(L, 4, 0);
  }

  tm_adj_init(L, Gi, graph);
  tm_add_seed_edges_immediate(L, graph);

  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, (int64_t)graph->uids->n);
    lua_pushinteger(L, tk_dsu_components(graph->dsu));
    lua_pushinteger(L, (int64_t) graph->n_edges);
    lua_pushstring(L, "seed");
    lua_call(L, 4, 0);
  }

  if (graph->category_inv && graph->category_anchors > 0) {
    uint64_t old_uid_count = graph->uids->n;
    tk_pvec_t *new_edges = tm_add_anchor_edges_immediate(L, Gi, graph);
    if (new_edges) {
      if (graph->uids->n > old_uid_count) {
        tk_dsu_add_ids(L, graph->dsu);
        tm_adj_resize(L, Gi, graph);
      }
      for (uint64_t i = 0; i < new_edges->n; i++) {
        int64_t u = new_edges->a[i].i;
        int64_t v = new_edges->a[i].p;
        tk_graph_add_adj(graph, u, v);
        tk_dsu_union(graph->dsu, u, v);
      }
      tk_pvec_destroy(new_edges);
      if (i_each != -1) {
        lua_pushvalue(L, i_each);
        lua_pushinteger(L, (int64_t)graph->uids->n);
        lua_pushinteger(L, tk_dsu_components(graph->dsu));
        lua_pushinteger(L, (int64_t) graph->n_edges);
        lua_pushstring(L, "anchors");
        lua_call(L, 4, 0);
      }
    }
  }

  if (graph->random_pairs > 0) {
    tk_pvec_t *pairs = NULL;
    int result = tk_graph_random_pairs(L, graph->uids, NULL, graph->random_pairs, &pairs);
    if (result != 0 || !pairs) {
      tk_lua_verror(L, 2, "random_pairs", "failed to generate random pairs");
      return 0;
    }
    int kha;
    uint32_t khi;
    for (uint64_t i = 0; i < pairs->n; i++) {
      int64_t u = pairs->a[i].i;
      int64_t v = pairs->a[i].p;
      uint32_t khi_u = tk_iumap_get(graph->uids_idx, u);
      uint32_t khi_v = tk_iumap_get(graph->uids_idx, v);
      if (khi_u == tk_iumap_end(graph->uids_idx) || khi_v == tk_iumap_end(graph->uids_idx))
        continue;
      tk_edge_t e = tk_edge(u, v, 0.0);
      khi = tk_euset_put(graph->pairs, e, &kha);
      if (!kha)
        continue;
      tk_graph_add_adj(graph, u, v);
      tk_dsu_union(graph->dsu, u, v);
      graph->n_edges++;
    }
    tk_pvec_destroy(pairs);
    if (i_each != -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, (int64_t)graph->uids->n);
      lua_pushinteger(L, tk_dsu_components(graph->dsu));
      lua_pushinteger(L, (int64_t) graph->n_edges);
      lua_pushstring(L, "random");
      lua_call(L, 4, 0);
    }
  }

  if (graph->knn) {
    uint64_t old_uid_count = graph->uids->n;
    tm_run_knn_queries(L, Gi, graph);
    if (graph->uids->n > old_uid_count) {
      tk_dsu_add_ids(L, graph->dsu);
      tm_adj_resize(L, Gi, graph);
    }
    tm_add_knn(L, graph);
    if (i_each != -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, (int64_t)graph->uids->n);
      lua_pushinteger(L, tk_dsu_components(graph->dsu));
      lua_pushinteger(L, (int64_t) graph->n_edges);
      lua_pushstring(L, "knn");
      lua_call(L, 4, 0);
    }
  }

  if (!graph->bridge && graph->knn && graph->dsu->components > 1) {

    tk_iumap_t *comp_sizes = tk_iumap_create(L, 0);
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
    lua_pop(L, 1);
    for (uint64_t i = 0; i < graph->uids->n; i++) {
      int64_t root = tk_dsu_findx(graph->dsu, (int64_t)i);
      tk_iumap_inc(comp_sizes, root);
    }
    int64_t max_size = 0;
    int64_t largest_root = -1;
    for (uint32_t k = tk_iumap_begin(comp_sizes); k != tk_iumap_end(comp_sizes); ++k) {
      if (!tk_iumap_exist(comp_sizes, k)) continue;
      int64_t size = tk_iumap_val(comp_sizes, k);
      if (size > max_size) {
        max_size = size;
        largest_root = tk_iumap_key(comp_sizes, k);
      }
    }
    graph->largest_component_root = largest_root;
    int64_t component_edges = 0;
    tk_edge_t p;
    tk_umap_foreach_keys(graph->pairs, p, ({
      uint32_t ukhi = tk_iumap_get(graph->uids_idx, p.u);
      uint32_t vkhi = tk_iumap_get(graph->uids_idx, p.v);
      if (ukhi != tk_iumap_end(graph->uids_idx) && vkhi != tk_iumap_end(graph->uids_idx)) {
        int64_t uidx = tk_iumap_val(graph->uids_idx, ukhi);
        int64_t vidx = tk_iumap_val(graph->uids_idx, vkhi);
        if (tk_dsu_findx(graph->dsu, uidx) == largest_root &&
            tk_dsu_findx(graph->dsu, vidx) == largest_root) {
          component_edges++;
        }
      }
    }))
    if (i_each != -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, max_size);
      lua_pushinteger(L, 1);
      lua_pushinteger(L, component_edges);
      lua_pushstring(L, "largest");
      lua_call(L, 4, 0);
    }

  } else if (graph->bridge && graph->knn_cache && graph->dsu->components > 1) {

    tk_evec_t *cs = tm_mst_knn_candidates(L, graph);
    if (cs == NULL)
      tk_lua_verror(L, 2, "graph_adj", "allocation failed during candidate collection");
    tm_add_mst(L, Gi, graph, cs);
    tk_evec_destroy(cs);
    if (i_each != -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, (int64_t)graph->uids->n);
      lua_pushinteger(L, tk_dsu_components(graph->dsu));
      lua_pushinteger(L, (int64_t) graph->n_edges);
      lua_pushstring(L, "kruskal");
      lua_call(L, 4, 0);
    }

  }

  if (graph->bridge && graph->dsu->components > 1) {
    tm_add_mst(L, Gi, graph, NULL);
    if (i_each != -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, (int64_t)graph->uids->n);
      lua_pushinteger(L, tk_dsu_components(graph->dsu));
      lua_pushinteger(L, (int64_t) graph->n_edges);
      lua_pushstring(L, "bridge");
      lua_call(L, 4, 0);
    }
  }

  tk_ivec_t *selected_nodes = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);
  tk_iumap_t *selected_set = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);
  int kha;
  uint32_t khi;

  if (graph->largest_component_root != -1) {
    for (uint64_t old_idx = 0; old_idx < graph->uids->n; old_idx++) {
      if (tk_dsu_findx(graph->dsu, (int64_t)old_idx) == graph->largest_component_root) {
        khi = tk_iumap_put(selected_set, (int64_t)old_idx, &kha);
        tk_iumap_setval(selected_set, khi, (int64_t)selected_nodes->n);
        if (tk_ivec_push(selected_nodes, (int64_t)old_idx) != 0) {
          tk_lua_verror(L, 2, "graph_adj", "allocation failed");
          return 0;
        }
      }
    }
  } else {
    for (uint64_t old_idx = 0; old_idx < graph->uids->n; old_idx++) {
      khi = tk_iumap_put(selected_set, (int64_t)old_idx, &kha);
      tk_iumap_setval(selected_set, khi, (int64_t)selected_nodes->n);
      if (tk_ivec_push(selected_nodes, (int64_t)old_idx) != 0) {
        tk_lua_verror(L, 2, "graph_adj", "allocation failed");
        return 0;
      }
    }
  }

  tm_compute_base_distances(L, Gi, graph);
  tm_compute_sigma(L, Gi, graph);
  tm_reweight_all_edges(L, graph);

  uint64_t n_nodes = selected_nodes->n;
  tk_ivec_t *tmp_offset = tk_ivec_create(L, n_nodes + 1, 0, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);
  tk_ivec_t *tmp_data = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);
  tk_dvec_t *tmp_weights = tk_dvec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);
  tmp_offset->a[0] = 0;
  for (uint64_t i = 0; i < n_nodes; i++) {
    int64_t old_idx = selected_nodes->a[i];
    int64_t u_uid = graph->uids->a[old_idx];
    tk_ivec_t *neighbors = tk_iuset_keys(L, graph->adj->a[old_idx]);
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
    lua_pop(L, 1);
    tk_rvec_t *neighbor_weights = tk_rvec_create(L, 0, 0, 0);
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
    lua_pop(L, 1);
    for (uint64_t j = 0; j < neighbors->n; j++) {
      int64_t neighbor_old_idx = neighbors->a[j];
      uint32_t khi = tk_iumap_get(selected_set, neighbor_old_idx);
      if (khi != tk_iumap_end(selected_set)) {
        int64_t neighbor_new_idx = tk_iumap_val(selected_set, khi);
        int64_t v_uid = graph->uids->a[neighbor_old_idx];
        double w = tk_graph_get_weight(graph, u_uid, v_uid);
        if (tk_rvec_push(neighbor_weights, tk_rank(neighbor_new_idx, w)) != 0) {
          tk_lua_verror(L, 2, "graph_adj", "allocation failed");
          return 0;
        }
      }
    }
    tk_rvec_desc(neighbor_weights, 0, neighbor_weights->n);
    for (uint64_t j = 0; j < neighbor_weights->n; j++) {
      if (tk_ivec_push(tmp_data, neighbor_weights->a[j].i) != 0) {
        tk_lua_verror(L, 2, "graph_adj", "allocation failed");
        return 0;
      }
      if (tk_dvec_push(tmp_weights, neighbor_weights->a[j].d) != 0) {
        tk_lua_verror(L, 2, "graph_adj", "allocation failed");
        return 0;
      }
    }
    tmp_offset->a[i + 1] = (int64_t) tmp_data->n;
  }
  tmp_offset->n = n_nodes + 1;

  tk_ivec_t *final_uids = tk_ivec_create(L, n_nodes, 0, 0);
  tk_ivec_t *final_offset = tk_ivec_create(L, n_nodes + 1, 0, 0);
  tk_ivec_t *final_data = tk_ivec_create(L, tmp_data->n, 0, 0);
  tk_dvec_t *final_weights = tk_dvec_create(L, tmp_weights->n, 0, 0);
  final_offset->a[0] = 0;
  int64_t write = 0;
  for (uint64_t i = 0; i < n_nodes; i++) {
    int64_t original_idx = selected_nodes->a[i];
    final_uids->a[i] = graph->uids->a[original_idx];
    int64_t edge_start = tmp_offset->a[i];
    int64_t edge_end = tmp_offset->a[i + 1];
    for (int64_t e = edge_start; e < edge_end; e++) {
      final_data->a[write] = tmp_data->a[e];
      final_weights->a[write] = tmp_weights->a[e];
      write++;
    }
    final_offset->a[i + 1] = write;
  }

  final_uids->n = n_nodes;
  final_offset->n = n_nodes + 1;
  final_data->n = tmp_data->n;
  final_weights->n = tmp_weights->n;

  tm_graph_destroy_internal(graph);
  tk_lua_replace(L, 1, 4);
  lua_settop(L, 4);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 4;
}

static inline tk_graph_t *tm_graph_create (
  lua_State *L,
  tk_pvec_t *edges,
  tk_inv_t *knn_inv,
  tk_ann_t *knn_ann,
  tk_hbi_t *knn_hbi,
  tk_ivec_sim_type_t knn_cmp,
  double knn_cmp_alpha,
  double knn_cmp_beta,
  int64_t knn_rank,
  tk_inv_t *category_inv,
  tk_ivec_sim_type_t category_cmp,
  double category_alpha,
  double category_beta,
  uint64_t category_anchors,
  uint64_t category_knn,
  double category_knn_decay,
  int64_t category_ranks,
  tk_inv_t *weight_inv,
  tk_ann_t *weight_ann,
  tk_hbi_t *weight_hbi,
  tk_ivec_sim_type_t weight_cmp,
  double weight_alpha,
  double weight_beta,
  tk_graph_weight_pooling_t weight_pooling,
  uint64_t random_pairs,
  double weight_eps,
  tk_graph_reweight_t reweight,
  int64_t sigma_k,
  double sigma_scale,
  uint64_t knn,
  uint64_t knn_cache,
  double knn_eps,
  bool bridge,
  uint64_t probe_radius
) {
  tk_graph_t *graph = tk_lua_newuserdata(L, tk_graph_t, TK_GRAPH_MT, NULL, tm_graph_gc);
  graph->edges = edges;
  graph->knn_inv = knn_inv;
  graph->knn_ann = knn_ann;
  graph->knn_hbi = knn_hbi;
  graph->knn_cmp = knn_cmp;
  graph->knn_cmp_alpha = knn_cmp_alpha;
  graph->knn_cmp_beta = knn_cmp_beta;
  graph->knn_rank = knn_rank;
  graph->knn_inv_hoods = NULL;
  graph->knn_ann_hoods = NULL;
  graph->knn_hbi_hoods = NULL;

  graph->category_inv = category_inv;
  graph->category_cmp = category_cmp;
  graph->category_alpha = category_alpha;
  graph->category_beta = category_beta;
  graph->category_anchors = category_anchors;
  graph->category_knn = category_knn;
  graph->category_knn_decay = category_knn_decay;
  graph->category_ranks = category_ranks;

  graph->weight_inv = weight_inv;
  graph->weight_ann = weight_ann;
  graph->weight_hbi = weight_hbi;
  graph->weight_cmp = weight_cmp;
  graph->weight_alpha = weight_alpha;
  graph->weight_beta = weight_beta;
  graph->weight_pooling = weight_pooling;

  graph->random_pairs = random_pairs;
  graph->weight_eps = weight_eps;
  graph->reweight = reweight;
  graph->sigma_k = sigma_k;
  graph->sigma_scale = sigma_scale;
  graph->knn = knn;
  graph->knn_cache = knn_cache;
  graph->knn_eps = knn_eps;
  graph->bridge = bridge;
  graph->probe_radius = probe_radius;
  graph->largest_component_root = -1;
  graph->pairs = tk_euset_create(L, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, lua_gettop(L), -1);
  lua_pop(L, 1);
  graph->n_edges = 0;
  graph->uids_hoods = NULL;
  graph->uids_idx_hoods = NULL;

  graph->q_weights = NULL;
  graph->e_weights = NULL;
  graph->inter_weights = NULL;
  if (weight_inv && weight_inv->ranks && weight_inv->n_ranks > 1) {
    graph->q_weights = tk_dvec_create(L, 0, 0, 0);
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, lua_gettop(L), -1);
    lua_pop(L, 1);
    graph->e_weights = tk_dvec_create(L, 0, 0, 0);
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, lua_gettop(L), -1);
    lua_pop(L, 1);
    graph->inter_weights = tk_dvec_create(L, 0, 0, 0);
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, lua_gettop(L), -1);
    lua_pop(L, 1);
  }

  graph->destroyed = false;
  return graph;
}

static inline int tm_adj_pairs (lua_State *L)
{
  lua_settop(L, 3);
  tk_ivec_t *ids = tk_ivec_peek(L, 1, "ids");
  tk_pvec_t *pos = tk_pvec_peek(L, 2, "pos");
  tk_pvec_t *neg = tk_pvec_peek(L, 3, "neg");

  tk_ivec_t *offsets = NULL;
  tk_ivec_t *neighbors = NULL;
  tk_dvec_t *weights = NULL;

  int result = tk_graph_pairs_to_csr(ids, pos, neg, &offsets, &neighbors, &weights);
  if (result != 0)
    tk_lua_verror(L, 2, "adj_pairs", "failed to build CSR adjacency");

  lua_settop(L, 1);
  tk_ivec_register(L, offsets);
  tk_ivec_register(L, neighbors);
  tk_dvec_register(L, weights);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 4;
}

static inline int tm_star_hoods (lua_State *L)
{
  lua_settop(L, 3);

  tk_ivec_t *ids = tk_ivec_peek(L, 1, "ids");
  tk_inv_hoods_t *inv_hoods = tk_inv_hoods_peekopt(L, 2);
  tk_ann_hoods_t *ann_hoods = tk_ann_hoods_peekopt(L, 2);
  tk_hbi_hoods_t *hbi_hoods = tk_hbi_hoods_peekopt(L, 2);
  if (!inv_hoods && !ann_hoods && !hbi_hoods)
    tk_lua_verror(L, 2, "star_hoods", "hoods must be tk_inv_hoods_t, tk_ann_hoods_t, or tk_hbi_hoods_t");
  uint64_t n_hoods = inv_hoods ? inv_hoods->n : ann_hoods ? ann_hoods->n : hbi_hoods->n;
  if (n_hoods != ids->n)
    tk_lua_verror(L, 2, "star_hoods", "hoods size must match ids size");
  tk_pvec_t *pairs = NULL;
  if (tk_graph_star_hoods(L, ids, inv_hoods, ann_hoods, hbi_hoods, &pairs) != 0)
    tk_lua_verror(L, 2, "star_hoods", "failed to convert hoods");
  lua_settop(L, 0);
  tk_pvec_register(L, pairs);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 1;
}

static inline int tm_adj_hoods(lua_State *L)
{
  lua_settop(L, 3);

  tk_ivec_t *ids = tk_ivec_peek(L, 1, "ids");

  tk_inv_hoods_t *inv_hoods = tk_inv_hoods_peekopt(L, 2);
  tk_ann_hoods_t *ann_hoods = tk_ann_hoods_peekopt(L, 2);
  tk_hbi_hoods_t *hbi_hoods = tk_hbi_hoods_peekopt(L, 2);

  if (!inv_hoods && !ann_hoods && !hbi_hoods)
    tk_lua_verror(L, 2, "adj_hoods", "hoods must be inv_hoods, ann_hoods, or hbi_hoods");

  uint64_t n_hoods = inv_hoods ? inv_hoods->n :
                     ann_hoods ? ann_hoods->n :
                     hbi_hoods->n;

  if (n_hoods != ids->n)
    tk_lua_verror(L, 2, "adj_hoods", "hoods size must match ids size");

  uint64_t features = tk_lua_foptunsigned(L, 3, "adj_hoods", "features", 0);
  if (features == 0)
    tk_lua_verror(L, 3, "adj_hoods", "features", "required");

  tk_ivec_t *offsets = NULL;
  tk_ivec_t *neighbors = NULL;
  tk_dvec_t *weights = NULL;

  if (tk_graph_adj_hoods(L, ids, inv_hoods, ann_hoods, hbi_hoods, features,
                         &offsets, &neighbors, &weights) != 0)
    tk_lua_verror(L, 2, "adj_hoods", "failed to convert hoods to adjacency");

  lua_settop(L, 0);
  tk_ivec_register(L, offsets);
  tk_ivec_register(L, neighbors);
  tk_dvec_register(L, weights);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 3;
}

static luaL_Reg tm_graph_fns[] =
{
  { "adjacency", tm_adjacency },
  { "adj_pairs", tm_adj_pairs },
  { "star_hoods", tm_star_hoods },
  { "adj_hoods", tm_adj_hoods },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_graph (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tm_graph_fns, 0);
  return 1;
}
