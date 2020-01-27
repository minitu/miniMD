/* ----------------------------------------------------------------------
   miniMD is a simple, parallel molecular dynamics (MD) code.   miniMD is
   an MD microapplication in the Mantevo project at Sandia National
   Laboratories ( http://www.mantevo.org ). The primary
   authors of miniMD are Steve Plimpton (sjplimp@sandia.gov) , Paul Crozier
   (pscrozi@sandia.gov) and Christian Trott (crtrott@sandia.gov).

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This library is free software; you
   can redistribute it and/or modify it under the terms of the GNU Lesser
   General Public License as published by the Free Software Foundation;
   either version 3 of the License, or (at your option) any later
   version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this software; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
   USA.  See also: http://www.gnu.org/licenses/lgpl.txt .

   For questions, contact Paul S. Crozier (pscrozi@sandia.gov) or
   Christian Trott (crtrott@sandia.gov).

   Please read the accompanying README and LICENSE files.
---------------------------------------------------------------------- */
//#define PRINTDEBUG(a) a
#define PRINTDEBUG(a)
#include "stdio.h"
#include "integrate.h"
#include "math.h"
#include "mpi.h"

#ifdef DUMPI_TRACE
#include <dumpi/libdumpi/libdumpi.h>
#endif

#include <cuda_profiler_api.h>

extern int pack_comm_count;
extern int unpack_comm_count;
extern int pack_comm_self_count;
extern int pack_reverse_count;
extern int unpack_reverse_count;

double integrate_times[2] = {0, 0};
double force_time = 0;
extern double comm_times[3];
extern double rev_comm_times[3];

Integrate::Integrate() {sort_every=20;}
Integrate::~Integrate() {}

void Integrate::setup()
{
  dtforce = 0.5 * dt;
}

void Integrate::initialIntegrate()
{
  Kokkos::parallel_for(Kokkos::RangePolicy<TagInitialIntegrate>(0,nlocal), *this);
}

KOKKOS_INLINE_FUNCTION
void Integrate::operator() (TagInitialIntegrate, const int& i) const {
  v(i,0) += dtforce * f(i,0);
  v(i,1) += dtforce * f(i,1);
  v(i,2) += dtforce * f(i,2);
  x(i,0) += dt * v(i,0);
  x(i,1) += dt * v(i,1);
  x(i,2) += dt * v(i,2);
}

void Integrate::finalIntegrate()
{
  Kokkos::parallel_for(Kokkos::RangePolicy<TagFinalIntegrate>(0,nlocal), *this);
}

KOKKOS_INLINE_FUNCTION
void Integrate::operator() (TagFinalIntegrate, const int& i) const {
  v(i,0) += dtforce * f(i,0);
  v(i,1) += dtforce * f(i,1);
  v(i,2) += dtforce * f(i,2);
}

void Integrate::run(Atom &atom, Force* force, Neighbor &neighbor,
                    Comm &comm, Thermo &thermo, Timer &timer)
{
  int i, n;

  comm.timer = &timer;
  timer.array[TIME_TEST] = 0.0;

  int check_safeexchange = comm.check_safeexchange;

  mass = atom.mass;
  dtforce = dtforce / mass;

    int next_sort = sort_every>0?sort_every:ntimes+1;

#ifdef DUMPI_TRACE
    libdumpi_enable_profiling();
#endif

    // Start CUDA profiler
    cudaProfilerStart();

    double start_time;

    for(n = 0; n < ntimes; n++) {

      Kokkos::fence();

      x = atom.x;
      v = atom.v;
      f = atom.f;
      xold = atom.xold;
      nlocal = atom.nlocal;

      start_time = MPI_Wtime();
      initialIntegrate();
      Kokkos::fence();
      integrate_times[0] += MPI_Wtime() - start_time;

      timer.stamp();

      if((n + 1) % neighbor.every) {

        comm.communicate(atom);
        timer.stamp(TIME_COMM);

      } else {
          if(check_safeexchange) {
              double d_max = 0;

              for(i = 0; i < atom.nlocal; i++) {
                double dx = (x(i,0) - xold(i,0));

                if(dx > atom.box.xprd) dx -= atom.box.xprd;

                if(dx < -atom.box.xprd) dx += atom.box.xprd;

                double dy = (x(i,1) - xold(i,1));

                if(dy > atom.box.yprd) dy -= atom.box.yprd;

                if(dy < -atom.box.yprd) dy += atom.box.yprd;

                double dz = (x(i,2) - xold(i,2));

                if(dz > atom.box.zprd) dz -= atom.box.zprd;

                if(dz < -atom.box.zprd) dz += atom.box.zprd;

                double d = dx * dx + dy * dy + dz * dz;

                if(d > d_max) d_max = d;
              }

              d_max = sqrt(d_max);

              if((d_max > atom.box.xhi - atom.box.xlo) || (d_max > atom.box.yhi - atom.box.ylo) || (d_max > atom.box.zhi - atom.box.zlo))
                printf("Warning: Atoms move further than your subdomain size, which will eventually cause lost atoms.\n"
                "Increase reneighboring frequency or choose a different processor grid\n"
                "Maximum move distance: %lf; Subdomain dimensions: %lf %lf %lf\n",
                d_max, atom.box.xhi - atom.box.xlo, atom.box.yhi - atom.box.ylo, atom.box.zhi - atom.box.zlo);

          }

          timer.stamp_extra_start();
          comm.exchange(atom);
          if(n+1>=next_sort) {
            atom.sort(neighbor);
            next_sort +=  sort_every;
          }
          comm.borders(atom);
          timer.stamp_extra_stop(TIME_TEST);
          timer.stamp(TIME_COMM);

        Kokkos::fence();

	Kokkos::Profiling::pushRegion("neighbor::build");
        neighbor.build(atom);
	Kokkos::Profiling::popRegion();

        timer.stamp(TIME_NEIGH);
      }

      Kokkos::Profiling::pushRegion("force");
      force->evflag = (n + 1) % thermo.nstat == 0;
      start_time = MPI_Wtime();
      force->compute(atom, neighbor, comm, comm.me);
      Kokkos::fence();
      force_time += MPI_Wtime() - start_time;
      Kokkos::Profiling::popRegion();

      timer.stamp(TIME_FORCE);

      if(neighbor.halfneigh && neighbor.ghost_newton) {
        comm.reverse_communicate(atom);

        timer.stamp(TIME_COMM);
      }

      v = atom.v;
      f = atom.f;
      nlocal = atom.nlocal;

      Kokkos::fence();

      start_time = MPI_Wtime();
      finalIntegrate();
      Kokkos::fence();
      integrate_times[1] += MPI_Wtime() - start_time;

      if(thermo.nstat) thermo.compute(n + 1, atom, neighbor, force, timer, comm);
    }

    int world_size;
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
      printf("[Atom] pack_comm: %d, unpack_comm: %d, pack_comm_self: %d, pack_reverse: %d, unpack_reverse: %d\n",
          pack_comm_count, unpack_comm_count, pack_comm_self_count, pack_reverse_count, unpack_reverse_count);
    }

    double* g_integrate_times = (double*)malloc(sizeof(double) * 2 * world_size);
    double* g_comm_times = (double*)malloc(sizeof(double) * 3 * world_size);
    double* g_force_times = (double*)malloc(sizeof(double) * world_size);
    double* g_rev_comm_times = (double*)malloc(sizeof(double) * 3 * world_size);
    MPI_Gather(integrate_times, 2, MPI_DOUBLE, g_integrate_times, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(comm_times, 3, MPI_DOUBLE, g_comm_times, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&force_time, 1, MPI_DOUBLE, g_force_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(rev_comm_times, 3, MPI_DOUBLE, g_rev_comm_times, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      // Average
      for (int i = 0; i < world_size; i++) {
        for (int j = 0; j < 2; j++) g_integrate_times[2*i+j] /= ntimes;
        for (int j = 0; j < 3; j++) g_comm_times[3*i+j] /= ntimes;
        g_force_times[i] /= ntimes;
        for (int j = 0; j < 3; j++) g_rev_comm_times[3*i+j] /= ntimes;
      }

      double max_comm_time = 0;
      int max_rank;

      for (int i = 0; i < world_size; i++) {
        if (g_comm_times[3*i+1] > max_comm_time) {
          max_rank = i;
          max_comm_time = g_comm_times[3*i+1];
        }
        printf("[Rank %d] inte[0]: %.3lf, comm[0]: %.3lf, comm[1]: %.3lf, comm[2]: %.3lf, force: %.3lf, "
            "rev_comm[0]: %.3lf, rev_comm[1]: %.3lf, rev_comm[2]: %.3lf, inte[1]: %.3lf\n",
            i, g_integrate_times[2*i] * 1000000, g_comm_times[3*i] * 1000000, g_comm_times[3*i+1] * 1000000, g_comm_times[3*i+2] * 1000000,
            g_force_times[i] * 1000000, g_rev_comm_times[3*i] * 1000000, g_rev_comm_times[3*i+1] * 1000000, g_rev_comm_times[3*i+2] * 1000000,
            g_integrate_times[2*i+1] * 1000000);
      }

      printf("[Max rank %d] inte[0]: %.3lf, comm[0]: %.3lf, comm[1]: %.3lf, comm[2]: %.3lf, force: %.3lf, "
          "rev_comm[0]: %.3lf, rev_comm[1]: %.3lf, rev_comm[2]: %.3lf, inte[1]: %.3lf\n",
          max_rank, g_integrate_times[2*max_rank] * 1000000, g_comm_times[3*max_rank] * 1000000, g_comm_times[3*max_rank+1] * 1000000, g_comm_times[3*max_rank+2] * 1000000,
          g_force_times[max_rank] * 1000000, g_rev_comm_times[3*max_rank] * 1000000, g_rev_comm_times[3*max_rank+1] * 1000000, g_rev_comm_times[3*max_rank+2] * 1000000,
          g_integrate_times[2*max_rank+1] * 1000000);
    }

    // Stop CUDA profiling
    cudaProfilerStop();

#ifdef DUMPI_TRACE
    libdumpi_disable_profiling();
#endif
}
