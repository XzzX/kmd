#include "kmd.h"

#include <iostream>

void interact(ParticleVector& particles,
              const double spring_constant,
              const double equilibrium_distance,
              const double cutoff_distance2)
{
    particles.pos_x.sync_device();
    particles.pos_y.sync_device();
    particles.force_x.sync_device();
    particles.force_y.sync_device();

    particles.force_x.modify_device();
    particles.force_y.modify_device();

    particles.s_force_x.reset();
    particles.s_force_y.reset();

    Kokkos::parallel_for("interact", NUM_PARTICLES, KOKKOS_LAMBDA (const int idx)
    {
        auto force_x = particles.s_force_x.access();
        auto force_y = particles.s_force_y.access();

        for (int idy = idx + 1; idy < NUM_PARTICLES; ++idy)
        {
            auto dx = particles.d_pos_x(idx) - particles.d_pos_x(idy);
            auto dy = particles.d_pos_y(idx) - particles.d_pos_y(idy);
            auto dist2 = dx*dx + dy*dy;
            if (dist2 < cutoff_distance2)
            {
                auto dist = std::sqrt(dist2);
                auto f = spring_constant * (dist - equilibrium_distance) / dist;

                force_x(idx) -= f * dx;
                force_y(idx) -= f * dy;

                force_x(idy) += f * dx;
                force_y(idy) += f * dy;
            }
        }
    });

    Kokkos::Experimental::contribute(particles.d_force_x, particles.s_force_x);
    Kokkos::Experimental::contribute(particles.d_force_y, particles.s_force_y);
}

void damp(ParticleVector& particles,
          const double alpha)
{
    particles.vel_x.sync_device();
    particles.vel_y.sync_device();
    particles.force_x.sync_device();
    particles.force_y.sync_device();

    particles.force_x.modify_device();
    particles.force_y.modify_device();

    Kokkos::parallel_for("damp", NUM_PARTICLES, KOKKOS_LAMBDA (const int idx)
    {
        particles.d_force_x(idx) -= alpha * particles.d_vel_x(idx);
        particles.d_force_y(idx) -= alpha * particles.d_vel_y(idx);
    });
}

void integrate(ParticleVector& particles,
               const double dt)
{
    particles.pos_x.sync_device();
    particles.pos_y.sync_device();
    particles.vel_x.sync_device();
    particles.vel_y.sync_device();
    particles.force_x.sync_device();
    particles.force_y.sync_device();

    particles.pos_x.modify_device();
    particles.pos_y.modify_device();
    particles.vel_x.modify_device();
    particles.vel_y.modify_device();
    particles.force_x.modify_device();
    particles.force_y.modify_device();

    Kokkos::parallel_for("integrate", NUM_PARTICLES, KOKKOS_LAMBDA (const int idx)
    {
        particles.d_vel_x(idx) += particles.d_force_x(idx) * dt;
        particles.d_vel_y(idx) += particles.d_force_y(idx) * dt;

        particles.d_pos_x(idx) += particles.d_vel_x(idx) * dt;
        particles.d_pos_y(idx) += particles.d_vel_y(idx) * dt;

        particles.d_force_x(idx) = 0.0;
        particles.d_force_y(idx) = 0.0;
    });
}

double dist(ParticleVector& particles,
            const size_t idx,
            const size_t idy)
{
    auto dx = particles.h_pos_x(idx) - particles.h_pos_x(idy);
    auto dy = particles.h_pos_y(idx) - particles.h_pos_y(idy);
    auto dist2 = dx*dx + dy*dy;
    return std::sqrt(dist2);
}