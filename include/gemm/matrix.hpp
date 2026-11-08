#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>

template <typename T>
concept Float32Or64 = std::same_as<T, float> || std::same_as<T, double>;

template <Float32Or64 FloatType> class MatrixHandler {
private:
  static constexpr std::size_t CACHE_LINE_SIZE{64};
  std::unique_ptr<FloatType> matrix;
  std::size_t column_padded{};
  int seed;
  std::size_t row;
  std::size_t column;

  void init_matrix(std::pair<FloatType, FloatType> fill_policy) {
    FloatType *_matrix{matrix.get()};
    std::random_device rd;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<FloatType> dist(fill_policy.first,
                                                   fill_policy.second);

    FloatType canary_float_NaN{std::numeric_limits<FloatType>::quiet_NaN()};

    for (int i{}; i < row; ++i) {
      for (int j{}; j < column_padded; ++j)
        if (j >= column) {
          _matrix[i * column_padded + j] = canary_float_NaN;
        } else {
          _matrix[i * column_padded + j] = dist(gen);
        }
    }
  }

public:
  MatrixHandler(std::size_t _row, std::size_t _column, int _seed,
                std::pair<FloatType, FloatType> fill_policy)
      : row(_row), column(_column), seed(_seed) {
    if (row == 0 || column == 0)
      throw std::runtime_error("rows and columns cannot be zero");

    std::size_t fp_bytes{sizeof(FloatType)};
    std::size_t cacheline_total_elements{CACHE_LINE_SIZE / fp_bytes};
    std::size_t column_multiple_up{column};

    if (column % cacheline_total_elements != 0) {
      column_multiple_up = column + (cacheline_total_elements -
                                     (column % cacheline_total_elements));
    }

    std::size_t total_elements_bytes{(row * column_multiple_up) * fp_bytes};

    void *_matrix{std::aligned_alloc(CACHE_LINE_SIZE, total_elements_bytes)};
    if (_matrix == nullptr)
      throw std::bad_alloc();

    matrix.reset(static_cast<FloatType *>(_matrix));
    column_padded = column_multiple_up;

    init_matrix(fill_policy);
  }

  MatrixHandler(const MatrixHandler &other) = delete;
  MatrixHandler(MatrixHandler &&other) noexcept = default;
  MatrixHandler &operator=(const MatrixHandler &other) = delete;
  MatrixHandler &operator=(MatrixHandler &&other) noexcept = default;

  const FloatType *data() { return matrix.get(); }

  std::tuple<std::size_t, std::size_t, std::size_t> dimensions() {
    return {row, column, column_padded};
  }
};
