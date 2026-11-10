#include "kernels.hpp"
#include "matrix.hpp"
#include <cstddef>
#include <gtest/gtest.h>

TEST(KernelTests, ExactTestSmallIntegers) {
  // Arrange
  std::size_t M{100};
  std::size_t N{70};
  std::size_t K{50};

  MatrixHandler<int> matrix_handler_A{M, K, 1000, {1, 4}};
  MatrixHandler<int> matrix_handler_B{K, N, 1000, {1, 4}};
  MatrixHandler<int> matrix_handler_C{M, N, 1000, {0, 0}};

  const int *matrix_A{matrix_handler_A.data()};
  auto [row_A, column_A, column_padded_A]{matrix_handler_A.dimensions()};

  const int *matrix_B{matrix_handler_B.data()};
  auto [row_B, column_B, column_padded_B]{matrix_handler_B.dimensions()};

  const int *const_matrix_C{matrix_handler_C.data()};
  auto [row_C, column_C, column_padded_C]{matrix_handler_C.dimensions()};
  int *matrix_C{const_cast<int *>(const_matrix_C)};

  // Act
  kernels::naive_int(matrix_A, matrix_B, matrix_C, M, N, K, column_padded_A,
                     column_padded_B, column_padded_C);

  for (int i{}; i < row_C; ++i) {
    for (int j{}; j < column_padded_C; ++j) {
      matrix_C[i * column_padded_C + j] =
          static_cast<float>(matrix_C[i * column_padded_C + j]);
    }
  }

  // Assert
  for (int i{}; i < M; ++i) {
    for (int j{}; j < N; ++j) {
      float expected{};
      for (int k{}; k < K; ++k) {
        expected += matrix_A[i * K + k] * matrix_B[k * N + j];
      }
      EXPECT_EQ(matrix_C[i * N + j], expected);
    }
  }
}
