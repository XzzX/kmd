add_executable(triangle
        kmd.cpp
        triangle.cpp)
target_compile_options(triangle PRIVATE
        -Wall
        -Wextra
        -Wunreachable-code
        -Wuninitialized
        -Wshadow)
target_link_libraries(triangle PRIVATE Kokkos::kokkos)
