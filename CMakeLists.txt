cmake_minimum_required(VERSION 3.12)
project(KMD CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Kokkos REQUIRED)

add_executable(kmd
        src/main.cpp)
target_compile_options(kmd PRIVATE
        -Wall
        -Wextra
        -Wunreachable-code
        -Wuninitialized
        -Wshadow)
target_link_libraries(kmd PRIVATE Kokkos::kokkos)
