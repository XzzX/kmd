#include "kmd.h"

#include <iostream>

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard kokkos(argc, argv);

    std::cout << "using kokkos execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    auto particles = init_particles();

    for (auto i = 0; i < 10000; ++i)
    {
        interact(particles);
        damp(particles);
        integrate(particles);
    }

    particles.sync_host_all();

//    for (size_t idx = 0; idx < NUM_PARTICLES; ++idx)
//    {
//        std::cout << idx << ": " << particles.h_pos_x(idx) << ", " << particles.h_pos_y(idx) << std::endl;
//    }

    for (size_t idx = 0; idx < NUM_PARTICLES; ++idx)
    {
        std::cout << idx << ": " << particles.h_pos_x(idx) << ", " << particles.h_pos_y(idx) << std::endl;
        std::cout << idx << ": " << particles.h_force_x(idx) << ", " << particles.h_force_y(idx) << std::endl;
    }

    std::cout << dist(particles, 0, 1) << std::endl;
    std::cout << dist(particles, 0, 2) << std::endl;
    std::cout << dist(particles, 1, 2) << std::endl;
}