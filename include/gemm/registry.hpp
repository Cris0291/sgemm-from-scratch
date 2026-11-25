#include "kernels.hpp"
#include <cstddef>
#include <string>

template <typename T>
using GemmFunc = void (*)(const T *A, const T *B, T *C, std::size_t M,
                          std::size_t N, std::size_t K, std::size_t lda,
                          std::size_t ldb, std::size_t ldc);

template <typename T> struct KernelEntry {
  std::string name{};
  GemmFunc<T> func;
};

struct Shapes {
  std::size_t M;
  std::size_t N;
  std::size_t K;
};

constexpr KernelEntry<float> kernels_float[]{{"naive", &kernels::naive<float>}};

constexpr KernelEntry<float> kernels_double[]{
    {"naive", &kernels::naive<float>}};

constexpr Shapes shapes[]{{20, 20, 20}};
