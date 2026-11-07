cmake_minimum_required(VERSION 3.20)
project(gemm-cpp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(gemm_headers INTERFACE)
target_include_directories(gemm_headers INTERFACE ${PROJECT_SOURCE_DIR}/include/gemm)

add_library(gemm_kernels src/kernels/naive.cpp)
target_link_libraries(gemm_kernels PUBLIC gemm_headers)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
)

FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(matrix_tests tests/test_matrix.cpp)
target_link_libraries(matrix_tests PRIVATE gemm_headers GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(matrix_tests)

