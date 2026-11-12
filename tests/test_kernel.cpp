#include "kernels.hpp"
#include "matrix.hpp"
#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>

TEST(KernelTests, ExactTestSmallIntegers) {
  // Arrange
  std::size_t M{100};
  std::size_t N{70};
  std::size_t K{50};

  MatrixHandler<float> matrix_handler_A{
      M, K, 1000, FillPolicy::IntegerValue, {1, 4}};
  MatrixHandler<float> matrix_handler_B{
      K, N, 1000, FillPolicy::IntegerValue, {1, 4}};
  MatrixHandler<float> matrix_handler_C{
      M, N, 1000, FillPolicy::IntegerValue, {2, 7}};

  const float *matrix_A{matrix_handler_A.data()};
  auto [row_A, column_A, column_padded_A]{matrix_handler_A.dimensions()};

  const float *matrix_B{matrix_handler_B.data()};
  auto [row_B, column_B, column_padded_B]{matrix_handler_B.dimensions()};

  float *matrix_C{matrix_handler_C.data()};
  auto [row_C, column_C, column_padded_C]{matrix_handler_C.dimensions()};

  std::vector<float> copy_matrix_C{};

  for (int i{}; i < row_C; ++i) {
    for (int j{}; j < column_padded_C; ++j) {
      copy_matrix_C.push_back(matrix_C[i * column_padded_C + j]);
    }
  }

  // Act
  kernels::naive_float(matrix_A, matrix_B, matrix_C, M, N, K, column_padded_A,
                       column_padded_B, column_padded_C);

  // Assert
  for (int i{}; i < M; ++i) {
    for (int j{}; j < N; ++j) {
      float expected{copy_matrix_C[i * column_padded_C + j]};
      for (int k{}; k < K; ++k) {
        expected += matrix_A[i * column_padded_A + k] *
                    matrix_B[k * column_padded_B + j];
      }
      EXPECT_FLOAT_EQ(matrix_C[i * column_padded_C + j], expected);
    }
  }
}
