#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_ScatterView.hpp>

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

void init_kmd();

void finalize_kmd();

void interact(ParticleVector& particles,
              const double spring_constant = 1.0,
              const double equilibrium_distance = 1.0,
              const double cutoff_distance2 = 4.0);

void damp(ParticleVector& particles,
          const double alpha = 0.1);

void integrate(ParticleVector& particles,
               const double dt = 0.1);


double dist(ParticleVector& particles,
            const size_t idx,
            const size_t idy);

ParticleVector init_particles();