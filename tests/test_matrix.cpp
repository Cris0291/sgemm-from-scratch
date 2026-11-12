#include "matrix.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

TEST(MatrixTests, PaddedCanariesTest) {
  // Arrange
  MatrixHandler<float> matrix_handler{
      100, 100, 1000, FillPolicy::RealValue, {0.0, 100.0}};
  const float *matrix{matrix_handler.data()};
  auto [row, column, column_padded]{matrix_handler.dimensions()};

  // Assert
  for (int i{}; i < row; ++i) {
    for (int j{}; j < column_padded; ++j) {
      if (j >= column) {
        EXPECT_TRUE(std::isnan(matrix[i * column_padded + j]));
      } else {
        EXPECT_FALSE(std::isnan(matrix[i * column_padded + j]));
      }
    }
  }
}

TEST(MatrixTests, PointerAligmentAndLdMultiple) {
  // Arrange
  MatrixHandler<float> matrix_handler{
      100, 100, 1000, FillPolicy::RealValue, {0.0, 100.0}};
  auto [row, column, column_padded]{matrix_handler.dimensions()};
  std::size_t CACHE_LINE_SIZE{64};
  std::size_t cacheline_total_elements{CACHE_LINE_SIZE / sizeof(float)};

  // Act
  const float *matrix{matrix_handler.data()};
  std::uintptr_t matrix_addr{reinterpret_cast<std::uintptr_t>(matrix)};

  // Assert
  EXPECT_TRUE(column_padded >= column);
  EXPECT_TRUE((column_padded % cacheline_total_elements) == 0);
  EXPECT_TRUE((matrix_addr & (CACHE_LINE_SIZE - 1)) == 0);
}
