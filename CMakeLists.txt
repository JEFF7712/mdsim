cmake_minimum_required(VERSION 3.20)
project(md)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenMP REQUIRED)

add_executable(main main.cpp)

target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)

if(MSVC)
    target_compile_options(main PRIVATE /O2 /arch:AVX2)
else()
    target_compile_options(main PRIVATE -O3 -march=native -ffast-math)
endif()