#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_ScatterView.hpp>

#include <iostream>

constexpr size_t NUM_PARTICLES = 3;

struct ParticleVector
{
    using ScalarView = Kokkos::DualView<double*>;
    using ScalarScatterView = Kokkos::Experimental::ScatterView<double*>;

    ScalarView pos_x = ScalarView("pos_x", NUM_PARTICLES);
    ScalarView pos_y = ScalarView("pos_y", NUM_PARTICLES);
    ScalarView vel_x = ScalarView("vel_x", NUM_PARTICLES);
    ScalarView vel_y = ScalarView("vel_y", NUM_PARTICLES);
    ScalarView force_x = ScalarView("force_x", NUM_PARTICLES);
    ScalarView force_y = ScalarView("force_y", NUM_PARTICLES);

    ScalarView::t_host h_pos_x = pos_x.view_host();
    ScalarView::t_host h_pos_y = pos_y.view_host();
    ScalarView::t_host h_vel_x = vel_x.view_host();
    ScalarView::t_host h_vel_y = vel_y.view_host();
    ScalarView::t_host h_force_x = force_x.view_host();
    ScalarView::t_host h_force_y = force_y.view_host();

    ScalarView::t_dev d_pos_x = pos_x.view_device();
    ScalarView::t_dev d_pos_y = pos_y.view_device();
    ScalarView::t_dev d_vel_x = vel_x.view_device();
    ScalarView::t_dev d_vel_y = vel_y.view_device();
    ScalarView::t_dev d_force_x = force_x.view_device();
    ScalarView::t_dev d_force_y = force_y.view_device();

    ScalarScatterView s_force_x = ScalarScatterView(d_force_x);
    ScalarScatterView s_force_y = ScalarScatterView(d_force_y);

    void sync_host_all()
    {
        pos_x.sync_host();
        pos_y.sync_host();

        vel_x.sync_host();
        vel_y.sync_host();

        force_x.sync_host();
        force_y.sync_host();
    }
};

void interact(ParticleVector& particles,
              const double spring_constant = 1.0,
              const double equilibrium_distance = 1.0,
              const double cutoff_distance2 = 4.0)
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

void damp(ParticleVector& particles, const double alpha = 0.1)
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

void integrate(ParticleVector& particles, const double dt = 0.1)
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

void init_particles(ParticleVector pv)
{
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
}

double dist(ParticleVector& particles, const size_t idx, const size_t idy)
{
    auto dx = particles.h_pos_x(idx) - particles.h_pos_x(idy);
    auto dy = particles.h_pos_y(idx) - particles.h_pos_y(idy);
    auto dist2 = dx*dx + dy*dy;
    return std::sqrt(dist2);
}

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard kokkos(argc, argv);
    std::cout << "using kokkos execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    ParticleVector particles;

    init_particles(particles);

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