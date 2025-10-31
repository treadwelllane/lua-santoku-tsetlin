#ifndef TK_ITQ_H
#define TK_ITQ_H

#include <santoku/iuset.h>
#include <santoku/cvec.h>
#include <santoku/dvec.h>
#include <santoku/dvec/ext.h>
#include <santoku/rvec/ext.h>
#include <santoku/iumap.h>
#include <santoku/dumap.h>
#include <santoku/pvec.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <lapacke.h>
#include <cblas.h>
#include <omp.h>

static inline void tk_itq_sign (
  char *out,
  double *X,
  uint64_t N,
  uint64_t K
) {
  #pragma omp parallel for
  for (uint64_t i = 0; i < N; i ++) {
    double *row = X + i * K;
    uint8_t *out_row = (uint8_t *)(out + i * TK_CVEC_BITS_BYTES(K));

    // Process complete bytes (8 bits at a time)
    uint64_t full_bytes = K / 8;
    for (uint64_t byte_idx = 0; byte_idx < full_bytes; byte_idx ++) {
      uint8_t byte_val = 0;
      uint64_t j_base = byte_idx * 8;
      for (uint64_t bit = 0; bit < 8; bit ++) {
        byte_val |= (row[j_base + bit] >= 0.0) << bit;
      }
      out_row[byte_idx] = byte_val;
    }

    // Handle remaining bits in final byte
    uint64_t remaining_start = full_bytes * 8;
    if (remaining_start < K) {
      uint8_t byte_val = 0;
      for (uint64_t j = remaining_start; j < K; j ++) {
        byte_val |= (row[j] >= 0.0) << (j - remaining_start);
      }
      out_row[full_bytes] = byte_val;
    }
  }
}

static inline void tk_itq_median (
  lua_State *L,
  char *out,
  double *X,
  uint64_t N,
  uint64_t K
) {
  #pragma omp parallel
  {
    double *col = malloc(N * sizeof(double));
    char *local_out = calloc(N * TK_CVEC_BITS_BYTES(K), 1);

    if (col && local_out) {
      #pragma omp for
      for (uint64_t j = 0; j < K; j ++) {
        // Extract column
        for (uint64_t i = 0; i < N; i ++)
          col[i] = X[i * K + j];

        // Find median
        ks_introsort(tk_dvec_asc, N, col);
        double med = (N & 1) ? col[N / 2] : 0.5 * (col[N / 2 - 1] + col[N / 2]);

        // Set bits in local buffer (no atomics needed)
        uint64_t byte_offset = TK_CVEC_BITS_BYTE(j);
        uint8_t bit_mask = (uint8_t) 1 << TK_CVEC_BITS_BIT(j);
        for (uint64_t i = 0; i < N; i ++) {
          if (X[i * K + j] >= med) {
            local_out[i * TK_CVEC_BITS_BYTES(K) + byte_offset] |= bit_mask;
          }
        }
      }

      // Merge local buffers into output
      #pragma omp critical
      {
        for (uint64_t idx = 0; idx < N * TK_CVEC_BITS_BYTES(K); idx ++) {
          out[idx] |= local_out[idx];
        }
      }

      free(local_out);
      free(col);
    }
  }
}

static inline void tk_itq_encode (
  lua_State *L,
  tk_dvec_t *codes,
  uint64_t n_dims,
  uint64_t max_iterations,
  double tolerance,
  int i_each
) {
  const uint64_t K = n_dims;
  const size_t N = codes->n / K;
  tk_cvec_t *out = tk_cvec_create(L, N * TK_CVEC_BITS_BYTES(n_dims), 0, 0);
  tk_cvec_zero(out);

  // Allocate all memory in one block for better cache locality
  size_t total_size = (N * K * 3 +  // X, V0, B
                       K * K * 4 +    // R, BtV, U, VT
                       K +            // S
                       K - 1) *       // superb
                      sizeof(double);
  double *mem = tk_malloc(L, total_size);

  double *X = mem;
  double *V0 = X + N * K;
  double *B = V0 + N * K;
  double *R = B + N * K;
  double *BtV = R + K * K;
  double *U = BtV + K * K;
  double *VT = U + K * K;
  double *S = VT + K * K;
  double *superb = S + K;

  memcpy(X, codes->a, N * K * sizeof(double));
  tk_dvec_center(X, N, K);

  // Initialize R to identity
  #pragma omp parallel for collapse(2)
  for (uint64_t i = 0; i < K; i ++)
    for (uint64_t j = 0; j < K; j ++)
      R[i * K + j] = (i == j ? 1.0 : 0.0);

  double last_obj = DBL_MAX, first_obj = 0.0;
  uint64_t it = 0;

  for (it = 0; it < max_iterations; it ++) {
    // V0 = X * R
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, K, K, 1.0, X, K, R, K, 0.0, V0, K);

    // Fuse: compute B = sign(V0) and objective = ||B - V0||^2
    double obj = 0.0;
    #pragma omp parallel for reduction(+:obj)
    for (size_t idx = 0; idx < N * K; idx ++) {
      double v = V0[idx];
      double b = (v >= 0.0 ? 1.0 : -1.0);
      B[idx] = b;
      double d = b - v;
      obj += d * d;
    }

    if (it == 0)
      first_obj = obj;

    // Check convergence before expensive SVD
    if (it > 0 && fabs(last_obj - obj) < tolerance * fabs(obj))
      break;

    last_obj = obj;

    // BtV = B^T * X
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, K, N, 1.0, B, K, X, K, 0.0, BtV, K);

    // SVD: BtV = U * S * VT
    int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', K, K, BtV, K, S, U, K, VT, K, superb);
    if (info != 0) {
      free(mem);
      luaL_error(L, "ITQ SVD failed to converge (info=%d)", info);
    }

    // R = VT^T * U^T
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, K, K, K, 1.0, VT, K, U, K, 0.0, R, K);
  }

  // Final projection: V0 = X * R (reuse V0 for output)
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, K, K, 1.0, X, K, R, K, 0.0, V0, K);
  tk_itq_sign(out->a, V0, N, K);

  free(mem);

  if (i_each >= 0) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, (int64_t) it);
    lua_pushnumber(L, first_obj);
    lua_pushnumber(L, last_obj);
    lua_call(L, 3, 0);
  }
}

#endif
