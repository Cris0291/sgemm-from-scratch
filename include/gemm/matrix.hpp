#include <concepts>
#include <cstddef>

template <typename T>
concept Float32Or64 = std::same_as<T, float> || std::same_as<T, double>;

class MatrixHandler {
  template <typename Size1, typename Size2, typename FLoatType>
    requires std::same_as<Size1, std::size_t> &&
             std::same_as<Size2, std::size_t> && Float32Or64<FLoatType>
  static FLoatType *matrix_creation(Size1 size1, Size2 size2, FLoatType ftype) {

  }
};
