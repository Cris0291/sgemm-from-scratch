#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>

template <typename T>
concept Float32Or64 = std::same_as<T, float> || std::same_as<T, double>;

enum class FillPolicy { IntegerValue, RealValue };

template <Float32Or64 Float> struct AlignedDeleter {
  void operator()(Float *ptr) { std::free(ptr); }
};

template <Float32Or64 FloatType> class MatrixHandler {
private:
  static constexpr std::size_t CACHE_LINE_SIZE{64};
  std::unique_ptr<FloatType, AlignedDeleter<FloatType>> matrix;
  std::size_t column_padded{};
  int seed;
  std::size_t row;
  std::size_t column;
  FillPolicy fill_policy;

  void init_matrix(FillPolicy policy,
                   std::pair<FloatType, FloatType> fill_range) {
    FloatType *_matrix{matrix.get()};
    std::mt19937 gen(seed);
    FloatType canary_float_NaN{std::numeric_limits<FloatType>::quiet_NaN()};

    auto func{create_policy(policy, gen, fill_range.first, fill_range.second)};

    for (int i{}; i < row; ++i) {
      for (int j{}; j < column_padded; ++j)
        if (j >= column) {
          _matrix[i * column_padded + j] = canary_float_NaN;
        } else {
          _matrix[i * column_padded + j] = func();
        }
    }
  }

  std::function<FloatType()> create_policy(FillPolicy policy, std::mt19937 &gen,
                                           FloatType min, FloatType max) {
    if (policy == FillPolicy::RealValue) {
      std::uniform_real_distribution<FloatType> dist(min, max);
      return [&gen, dist]() mutable -> FloatType { return dist(gen); };
    } else {
      std::uniform_int_distribution<int> dist(min, max);
      return [&gen, dist]() mutable -> FloatType { return dist(gen); };
    }
  }

public:
  MatrixHandler(std::size_t _row, std::size_t _column, int _seed,
                FillPolicy _fill_policy,
                std::pair<FloatType, FloatType> fill_range)
      : row(_row), column(_column), seed(_seed), fill_policy(_fill_policy) {
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

    init_matrix(fill_policy, fill_range);
  }

  MatrixHandler(const MatrixHandler &other) = delete;
  MatrixHandler(MatrixHandler &&other) noexcept = default;
  MatrixHandler &operator=(const MatrixHandler &other) = delete;
  MatrixHandler &operator=(MatrixHandler &&other) noexcept = default;

  const FloatType *data() const { return matrix.get(); }

  FloatType *data() {
    return const_cast<FloatType *>(
        static_cast<const MatrixHandler &>(*this).matrix.get());
  }

  FloatType *data_non() { return matrix.get(); }

  std::tuple<std::size_t, std::size_t, std::size_t> dimensions() {
    return {row, column, column_padded};
  }
};
