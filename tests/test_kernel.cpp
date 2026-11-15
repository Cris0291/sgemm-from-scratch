#include "kernels.hpp"
#include "matrix.hpp"
#include <cstddef>
#include <gtest/gtest.h>

TEST(KernelTests, ExactTestSmallIntegers) {
  // Arrange
  std::size_t M{100};
  std::size_t N{70};
  std::size_t K{50};

  MatrixHandler<float> matrix_handler_A{
      M, K, 1000, FillPolicy::IntegerValue, {1, 4}, PaddingPolicy::FloatZero};
  MatrixHandler<float> matrix_handler_B{
      K, N, 700, FillPolicy::IntegerValue, {1, 4}, PaddingPolicy::FloatZero};
  MatrixHandler<float> matrix_handler_C{
      M, N, 10, FillPolicy::IntegerValue, {2, 7}, PaddingPolicy::FloatZero};

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
      EXPECT_EQ(matrix_C[i * column_padded_C + j], expected);
    }
  }

  for (int i{}; i < row_C; ++i) {
    for (int j{}; j < column_padded_C; ++j) {
      if (j >= column_C) {
        EXPECT_EQ(matrix_C[i * column_padded_C + j],
                  copy_matrix_C[i * column_padded_C + j]);

        EXPECT_EQ(matrix_C[i * column_padded_C + j], 0);

        EXPECT_EQ(copy_matrix_C[i * column_padded_C + j], 0);
      }
    }
  }
}

TEST(KernelTests, ToleranceFP64) {
  // Arrange
  std::size_t M{100};
  std::size_t N{70};
  std::size_t K{50};

  MatrixHandler<float> matrix_handler_A{
      M, K, 1000, FillPolicy::RealValue, {4.5, 10.7}, PaddingPolicy::FloatNaN};
  MatrixHandler<float> matrix_handler_B{
      K, N, 700, FillPolicy::RealValue, {11.3, 41.4}, PaddingPolicy::FloatNaN};
  MatrixHandler<float> matrix_handler_C{
      M, N, 10, FillPolicy::RealValue, {2.8, 7.9}, PaddingPolicy::FloatNaN};

  const float *matrix_A{matrix_handler_A.data()};
  auto [row_A, column_A, column_padded_A]{matrix_handler_A.dimensions()};

  const float *matrix_B{matrix_handler_B.data()};
  auto [row_B, column_B, column_padded_B]{matrix_handler_B.dimensions()};

  float *matrix_C{matrix_handler_C.data()};
  auto [row_C, column_C, column_padded_C]{matrix_handler_C.dimensions()};

  std::vector<double> fp64_matrix_C{};

  for (int i{}; i < M; ++i) {
    for (int j{}; j < N; ++j) {
      float expected{matrix_C[i * column_padded_C + j]};
      for (int k{}; k < K; ++k) {
        expected += matrix_A[i * column_padded_A + k] *
                    matrix_B[k * column_padded_B + j];
      }
      fp64_matrix_C.push_back(expected);
    }
  }
}
