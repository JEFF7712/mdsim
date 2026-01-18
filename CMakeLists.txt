cmake_minimum_required(VERSION 3.20)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
project(MD_Simulation LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED)

add_library(gpu_lib OBJECT src/gpu_interface.cu)
target_include_directories(gpu_lib PRIVATE include)
target_compile_options(gpu_lib PRIVATE -O3)
set_property(TARGET gpu_lib PROPERTY CUDA_ARCHITECTURES 86)
set_property(TARGET gpu_lib PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(main src/main.cpp)

set_property(TARGET main PROPERTY CUDA_ARCHITECTURES 86)

target_link_libraries(main PRIVATE gpu_lib OpenMP::OpenMP_CXX)
target_include_directories(main PRIVATE include)
target_compile_options(main PRIVATE -O3 -march=native -fopenmp)