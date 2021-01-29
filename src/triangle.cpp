#include "kmd.h"

#include <iostream>

ParticleVector init_particles()
{
    ParticleVector pv;

    for (size_t idx = 0; idx < NUM_PARTICLES; ++idx)
    {
        pv.h_pos_x(idx) = 0.0;
        pv.h_pos_y(idx) = 0.0;

        pv.h_vel_x(idx) = 0.0;
        pv.h_vel_y(idx) = 0.0;

        pv.h_force_x(idx) = 0.0;
        pv.h_force_y(idx) = 0.0;
    }

    pv.h_pos_x(0) = 0.0;
    pv.h_pos_y(0) = 0.0;

    pv.h_pos_x(1) = 1.5;
    pv.h_pos_y(1) = 0.0;

    pv.h_pos_x(2) = 0.75;
    pv.h_pos_y(2) = std::sqrt(1.5*1.5 - 0.75*0.75);

    pv.pos_x.modify_host();
    pv.pos_y.modify_host();
    pv.vel_x.modify_host();
    pv.vel_y.modify_host();
    pv.force_x.modify_host();
    pv.force_y.modify_host();

    return pv;
}

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