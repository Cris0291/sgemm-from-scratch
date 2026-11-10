#include "kernels.hpp"

void kernels::naive_float(const float *A, const float *B, float *C,
                          std::size_t M, std::size_t N, std::size_t K,
                          std::size_t lda, std::size_t ldb, std::size_t ldc) {
  for (int i{}; i < M; ++i) {
    for (int j{}; j < N; ++j) {
      for (int k{}; k < K; ++k) {
        C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
      }
    }
  }
}

void kernels::naive_int(const int *A, const int *B, int *C, std::size_t M,
                        std::size_t N, std::size_t K, std::size_t lda,
                        std::size_t ldb, std::size_t ldc) {
  for (int i{}; i < M; ++i) {
    for (int j{}; j < N; ++j) {
      for (int k{}; k < K; ++k) {
        C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
      }
    }
  }
}
