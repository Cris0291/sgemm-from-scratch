#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>

template <typename T>
concept Float32Or64 = std::same_as<T, float> || std::same_as<T, double>;

template <typename N>
concept PositiveNumber = std::integral<N> && !std::signed_integral<N>;

template <Float32Or64 FloatType, PositiveNumber RowSize,
          PositiveNumber ColumnSize>
class MatrixHandler {
private:
  static constexpr std::size_t CACHE_LINE_SIZE{64};
  FloatType *matrix;
  std::size_t column_padded{};
  int seed;
  RowSize row;
  ColumnSize column;

public:
  MatrixHandler(RowSize _row, ColumnSize _column, int _seed)
      : row(_row), column(_column), seed(_seed) {
    if (row == 0 || column == 0)
      throw std::runtime_error("rows and columns cannot be zero");
  }
  ~MatrixHandler() { free(static_cast<void *>(matrix)); }

  void matrix_creation(RowSize row_size, ColumnSize column_size,
                       FloatType ftype) {
    std::size_t fp_bytes{sizeof(FloatType)};
    std::size_t cacheline_total_elements{CACHE_LINE_SIZE / fp_bytes};
    std::size_t column_multiple_up{column_size};

    if (column_size % cacheline_total_elements != 0) {
      column_multiple_up =
          column_size +
          (cacheline_total_elements - (column_size % cacheline_total_elements));
    }

    std::size_t total_elements_bytes{(row_size * column_multiple_up) *
                                     fp_bytes};

    void *_matrix{std::aligned_alloc(CACHE_LINE_SIZE, total_elements_bytes)};
    matrix = static_cast<FloatType *>(_matrix);
    column_padded = column_multiple_up;
  }

  void init_matrix(FloatType *matrix, RowSize row_size,
                   ColumnSize column_size) {
    std::random_device rd;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<FloatType> dist(0.0, 100.0);

    FloatType canary_float_NaN{std::numeric_limits<FloatType>::quiet_NaN()};

    for (int i{}; i < row; ++i) {
      for (int j{}; j < column_padded; ++j)
        if (j >= column) {
          matrix[i * column_padded + j] = canary_float_NaN;
        } else {
          matrix[i * column_padded + j] = dist(gen);
        }
    }
  }
};
