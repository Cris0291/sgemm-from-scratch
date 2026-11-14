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

enum class PaddingPolicy { FloatNaN, FloatZero, FloatSentinel };

template <Float32Or64 Float> struct AlignedDeleter {
  void operator()(Float *ptr) const { std::free(ptr); }
};

struct MatrixDimensions {
  std::size_t row;
  std::size_t column;
  std::size_t column_padded;
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
  PaddingPolicy padding_policy;
  FloatType sentinel;
  std::pair<FloatType, FloatType> fill_range;

  void init_matrix() {
    FloatType *_matrix{matrix.get()};
    std::mt19937 gen(seed);
    FloatType matrix_padding{get_matrix_padding()};

    auto func{
        create_policy(fill_policy, gen, fill_range.first, fill_range.second)};

    for (int i{}; i < row; ++i) {
      for (int j{}; j < column_padded; ++j)
        if (j >= column) {
          _matrix[i * column_padded + j] = matrix_padding;
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

  FloatType get_matrix_padding() {
    FloatType res{};
    switch (padding_policy) {
    case PaddingPolicy::FloatNaN: {
      FloatType canary_float_NaN{std::numeric_limits<FloatType>::quiet_NaN()};
      res = canary_float_NaN;
      break;
    }
    case PaddingPolicy::FloatZero: {
      break;
    }
    case PaddingPolicy::FloatSentinel: {
      res = sentinel;
      break;
    }
    }

    return res;
  }

public:
  MatrixHandler(std::size_t _row, std::size_t _column, int _seed,
                FillPolicy _fill_policy,
                std::pair<FloatType, FloatType> _fill_range,
                PaddingPolicy _padding_policy, FloatType _sentinel = 0)
      : row(_row), column(_column), seed(_seed), fill_policy(_fill_policy),
        padding_policy(_padding_policy), sentinel(_sentinel),
        fill_range(_fill_range) {
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

    init_matrix();
  }

  MatrixHandler(const MatrixHandler &other) = delete;
  MatrixHandler(MatrixHandler &&other) noexcept = default;
  MatrixHandler &operator=(const MatrixHandler &other) = delete;
  MatrixHandler &operator=(MatrixHandler &&other) noexcept = default;

  const FloatType *data() const { return matrix.get(); }

  FloatType *data() { return matrix.get(); }

  std::tuple<std::size_t, std::size_t, std::size_t> dimensions() {
    return {row, column, column_padded};
  }
};
