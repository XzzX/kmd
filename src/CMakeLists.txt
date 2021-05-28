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

pybind11_add_module(kmd kmd.python.cpp kmd.cpp)
target_link_libraries(kmd PRIVATE Kokkos::kokkos)
set_target_properties(kmd PROPERTIES POSITION_INDEPENDENT_CODE ON)