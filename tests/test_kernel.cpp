#include "kernels.hpp"
#include "matrix.hpp"
#include <gtest/gtest.h>

TEST(KernelTests, ExactTestSmallIntegers) {
  // Arrange
  MatrixHandler<int> matrix_handler_A{100, 50, 1000, {0, 100}};
  MatrixHandler<int> matrix_handler_B{50, 70, 1000, {0, 100}};
  const int *matrix_A{matrix_handler_A.data()};
  auto [row_A, column_A, column_padded_A]{matrix_handler_A.dimensions()};
  const int *matrix_B{matrix_handler_A.data()};
  auto [row_B, column_B, column_padded_B]{matrix_handler_B.dimensions()};

  // Act
  kernels::naive_float(const float *A, const float *B, float *C, std::size_t M,
                       std::size_t N, std::size_t K, std::size_t lda,
                       std::size_t ldb, std::size_t ldc)
}
