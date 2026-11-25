#include <cstddef>

namespace kernels {
template <typename T>
void naive(const T *A, const T *B, T *C, std::size_t M, std::size_t N,
           std::size_t K, std::size_t lda, std::size_t ldb, std::size_t ldc);
}; // namespace kernels
