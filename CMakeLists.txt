cmake_minimum_required(VERSION 3.20)
project(gemm-cpp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(gemm_kernels src/kernels/naive.cpp)
target_include_directories(gemm_kernels PUBLIC ${PROJECT_SOURCE_DIR}/include/gemm)

