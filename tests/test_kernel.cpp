#include "kernels.hpp"
#include "matrix.hpp"
#include "gtest/gtest.h"
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>

class KernelTests : public ::testing::TestWithParam<int> {};

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

TEST(KernelTests, ToleranceFP64FixedTol) {
  // Arrange

  double atol{1e-5};
  double rtol{1e-4};
  double total_tol{atol + rtol};

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

  for (int i{}; i < row_C; ++i) {
    for (int j{}; j < column_padded_C; ++j) {
      fp64_matrix_C.push_back(
          static_cast<double>(matrix_C[i * column_padded_C + j]));
    }
  }

  for (int i{}; i < M; ++i) {
    for (int j{}; j < N; ++j) {
      for (int k{}; k < K; ++k) {
        fp64_matrix_C[i * column_padded_C + j] +=
            static_cast<double>(matrix_A[i * column_padded_A + k]) *
            static_cast<double>(matrix_B[k * column_padded_B + j]);
      }
    }
  }

  // Act
  kernels::naive_float(matrix_A, matrix_B, matrix_C, M, N, K, column_padded_A,
                       column_padded_B, column_padded_C);

  // Assert
  for (int i{}; i < row_C; ++i) {
    for (int j{}; j < column_padded_C; ++j) {
      if (j < column_C) {
        std::cout << "c 32 : " << matrix_C[i * column_padded_C + j]
                  << "c fp64 : " << fp64_matrix_C[i * column_padded_C + j]
                  << "\n";
        auto abs_res{std::abs(matrix_C[i * column_padded_C + j] -
                              fp64_matrix_C[i * column_padded_C + j])};
        EXPECT_LE(std::abs(abs_res), total_tol);
      }
    }
  }
}

TEST(KernelTests, SubBlock) {
  // Arrange
  std::size_t M{15};
  std::size_t N{9};
  std::size_t K{10};

  std::size_t sub_M{4};
  std::size_t sub_N{7};
  std::size_t sub_K{3};

  std::size_t init_x_A{2};
  std::size_t init_y_A{2};

  std::size_t init_x_B{3};
  std::size_t init_y_B{3};

  std::size_t init_x_C{3};
  std::size_t init_y_C{1};

  MatrixHandler<float> matrix_handler_A{
      M, K, 1000, FillPolicy::IntegerValue, {1, 4}, PaddingPolicy::FloatNaN,
  };
  MatrixHandler<float> matrix_handler_B{
      K, N, 700, FillPolicy::IntegerValue, {1, 4}, PaddingPolicy::FloatNaN,
  };
  MatrixHandler<float> matrix_handler_C{
      M, N, 10, FillPolicy::IntegerValue, {2, 7}, PaddingPolicy::FloatNaN,
  };

  const float *matrix_A{matrix_handler_A.data()};
  auto [row_A, column_A, column_padded_A]{matrix_handler_A.dimensions()};

  const float *matrix_B{matrix_handler_B.data()};
  auto [row_B, column_B, column_padded_B]{matrix_handler_B.dimensions()};

  float *matrix_C{matrix_handler_C.data()};
  auto [row_C, column_C, column_padded_C]{matrix_handler_C.dimensions()};

  auto [sub_matrix_A, sub_row_A, sub_column_A]{
      matrix_handler_A.create_submatrix(init_x_A, init_y_A, sub_M, sub_K)};

  auto [sub_matrix_B, sub_row_B, sub_column_B]{
      matrix_handler_B.create_submatrix(init_x_B, init_y_B, sub_K, sub_N)};

  auto [sub_matrix_C, sub_row_C, sub_column_C]{
      matrix_handler_C.create_submatrix(init_x_C, init_y_C, sub_M, sub_N)};

  std::vector<float> copy_matrix_C{};

  for (int i{}; i < row_C; ++i) {
    for (int j{}; j < column_padded_C; ++j) {
      copy_matrix_C.push_back(matrix_C[i * column_padded_C + j]);
    }
  }

  // Act
  kernels::naive_float(sub_matrix_A, sub_matrix_B, sub_matrix_C, sub_M, sub_N,
                       sub_K, column_padded_A, column_padded_B,
                       column_padded_C);

  // Assert
  for (int i{}; i < row_C; ++i) {
    for (int j{}; j < column_padded_C; ++j) {
      if (j >= column_C) {
        EXPECT_TRUE(std::isnan(copy_matrix_C[i * column_padded_C + j]));
      } else {
        if ((i >= init_x_C && i < init_x_C + sub_row_C) &&
            (j >= init_y_C && j < init_y_C + sub_column_C)) {
          EXPECT_EQ(copy_matrix_C[i * column_padded_C + j],
                    sub_matrix_C[i * column_padded_C + j]);
        } else {
          EXPECT_NE(matrix_C[i * column_padded_C + j],
                    sub_matrix_C[i * column_padded_C + j]);
        }
      }
    }
  }
}
