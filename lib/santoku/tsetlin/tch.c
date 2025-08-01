#include <santoku/tsetlin/conf.h>
#include <santoku/tsetlin/tch.h>
#include <santoku/tsetlin/graph.h>
#include <santoku/ivec.h>

#include <float.h>
#include <lauxlib.h>
#include <lua.h>

static inline int tk_tch_refine_lua (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "codes");
  tk_ivec_t *codes = tk_ivec_peek(L, -1, "codes");
  int i_out = tk_lua_absindex(L, -1);

  lua_getfield(L, 1, "scale");
  tk_dvec_t *scale = tk_dvec_peekopt(L, -1);

  lua_getfield(L, 1, "graph");
  tk_graph_t *graph = tk_graph_peek(L, -1);

  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "tch", "n_dims");
  double neg_scale = tk_lua_foptnumber(L, 1, "tch", "negatives", 1.0);

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  // Run tch
  tk_tch_refine(L, codes, scale, graph, n_dims, neg_scale, i_each);
  lua_pushvalue(L, i_out);
  return 1;
}

static luaL_Reg tk_tch_fns[] =
{
  { "refine", tk_tch_refine_lua },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_tch (lua_State *L)
{
  lua_newtable(L); // t
  tk_lua_register(L, tk_tch_fns, 0); // t
  return 1;
}
