#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <utility>

template <typename T>
concept Float32Or64 = std::same_as<T, float> || std::same_as<T, double>;

class MatrixHandler {
private:
  static constexpr std::size_t CACHE_LINE_SIZE{64};

public:
  template <typename RowSize, typename ColumnSize, typename FLoatType>
    requires std::same_as<RowSize, std::size_t> &&
             std::same_as<ColumnSize, std::size_t> && Float32Or64<FLoatType>
  static std::pair<FLoatType *, std::size_t>
  matrix_creation(RowSize row_size, ColumnSize column_size, FLoatType ftype) {
    std::size_t fp_bytes{sizeof(FLoatType)};
    std::size_t cacheline_total_elements{CACHE_LINE_SIZE / fp_bytes};
    std::size_t row_multiple_up{row_size};

    if (row_size % cacheline_total_elements != 0) {
      row_multiple_up = row_size + (cacheline_total_elements -
                                    (row_size % cacheline_total_elements));
    }

    std::size_t total_elements_bytes{(row_multiple_up * column_size) *
                                     fp_bytes};

    void *matrix{std::aligned_alloc(CACHE_LINE_SIZE, total_elements_bytes)};
  }

  static void init_matrix() {}
};
