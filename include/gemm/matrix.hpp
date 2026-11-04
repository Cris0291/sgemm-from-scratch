#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <random>
#include <utility>

template <typename T>
concept Float32Or64 = std::same_as<T, float> || std::same_as<T, double>;

class MatrixHandler {
private:
  static constexpr std::size_t CACHE_LINE_SIZE{64};

public:
  template <typename RowSize, typename ColumnSize, typename FloatType>
    requires std::same_as<RowSize, std::size_t> &&
             std::same_as<ColumnSize, std::size_t> && Float32Or64<FloatType>
  static std::pair<FloatType *, std::size_t>
  matrix_creation(RowSize row_size, ColumnSize column_size, FloatType ftype) {
    std::size_t fp_bytes{sizeof(FloatType)};
    std::size_t cacheline_total_elements{CACHE_LINE_SIZE / fp_bytes};
    std::size_t row_multiple_up{row_size};

    if (row_size % cacheline_total_elements != 0) {
      row_multiple_up = row_size + (cacheline_total_elements -
                                    (row_size % cacheline_total_elements));
    }

    std::size_t total_elements_bytes{(row_multiple_up * column_size) *
                                     fp_bytes};

    void *matrix{std::aligned_alloc(CACHE_LINE_SIZE, total_elements_bytes)};
    return {matrix, row_multiple_up};
  }

  template <typename RowSize, typename ColumnSize, typename FloatType>
    requires std::same_as<RowSize, std::size_t> &&
             std::same_as<ColumnSize, std::size_t> && Float32Or64<FloatType>

  static void init_matrix(FloatType *matrix, RowSize row_size,
                          ColumnSize column_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<FloatType> dist(0.0, 100.0);

    std::size_t total_elements{row_size * column_size};
    for (int i{}; i < total_elements; ++i) {
      matrix[i] = dist(gen);
    }
  }
};
