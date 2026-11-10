#include <cstddef>

namespace kernels {
void naive_float(const float *A, const float *B, float *C, std::size_t M,
                 std::size_t N, std::size_t K, std::size_t lda, std::size_t ldb,
                 std::size_t ldc);
}; // namespace kernels
