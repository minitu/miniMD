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
#include "hapi.h"
#include "hapi_nvtx.h"
#include <sstream>

Integrate::Integrate() {sort_every=20;}
Integrate::~Integrate() {}

void Integrate::setup()
{
  dtforce = 0.5 * dt;
}

void Integrate::initialIntegrate()
{
  std::ostringstream os;
  os << "Integrate::initialIntegrate " << index;
  NVTXTracer(os.str(), NVTXColor::Turquoise);
  // Should be invoked as separate kernels because of the dependency on v
  Kokkos::parallel_for(Kokkos::Experimental::require(
        Kokkos::RangePolicy<TagInitialIntegrate>(compute_instance,0,nlocal),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight), *this);
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
  std::ostringstream os;
  os << "Integrate::finalIntegrate " << index;
  NVTXTracer(os.str(), NVTXColor::Turquoise);
  Kokkos::parallel_for(Kokkos::Experimental::require(
        Kokkos::RangePolicy<TagFinalIntegrate>(compute_instance,0,nlocal),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight), *this);
}

KOKKOS_INLINE_FUNCTION
void Integrate::operator() (TagFinalIntegrate, const int& i) const {
  v(i,0) += dtforce * f(i,0);
  v(i,1) += dtforce * f(i,1);
  v(i,2) += dtforce * f(i,2);
}

void Integrate::run(Atom &atom, Force* force, Neighbor &neighbor,
                    Comm* comm, Thermo &thermo, int index)
{
  int i, n;

  int check_safeexchange = comm->check_safeexchange;

  mass = atom.mass;
  dtforce = dtforce / mass;

    int next_sort = sort_every>0?sort_every:ntimes+1;

    double total_time = 0;
    for(n = 0; n < ntimes; n++) {
      double iter_start_time = CkWallTimer();
      if (index == 0 && (n == 0 || n % 10 == 0)) {
        CkPrintf("[Block] Starting iteration %d\n", n);
      }

      // Store iteration counter in Comm
      comm->iter = n;

      x = atom.x;
      v = atom.v;
      f = atom.f;
      xold = atom.xold;
      nlocal = atom.nlocal;

      initialIntegrate();

      if((n + 1) % neighbor.every) {

        comm->communicate(atom, false);

      } else {
        // TODO: Reneighboring not supported (not converted to async)
        /*
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

        comm->exchange(atom, false);
        if(n+1>=next_sort) {
          atom.sort(neighbor);
          next_sort +=  sort_every;
        }
        comm->borders(atom, false);

        Kokkos::fence();

        Kokkos::Profiling::pushRegion("neighbor::build");
        neighbor.build(atom);
        Kokkos::Profiling::popRegion();
        */
      }

      Kokkos::Profiling::pushRegion("force");
      force->evflag = (n + 1) % thermo.nstat == 0;
      force->compute(atom, neighbor, comm, comm->index);
      Kokkos::Profiling::popRegion();

      if (neighbor.halfneigh && neighbor.ghost_newton) {
        comm->reverse_communicate(atom, false);
      }

      v = atom.v;
      f = atom.f;
      nlocal = atom.nlocal;

      finalIntegrate();

      if(thermo.nstat) thermo.compute(n + 1, atom, neighbor, force, comm);

      /*
      if (index == 0) {
        CkPrintf("[Block] Iteration %d time: %.6lf\n", n, CkWallTimer() - iter_start_time);
      }
      */

      // Don't include first iteration time
      if (n > 0) {
        total_time += CkWallTimer() - iter_start_time;
      }
    }

    if (index == 0) {
      CkPrintf("[Block] Total time (exclude 1st iteration): %.6lf s\n", total_time);
      CkPrintf("[Block] Average time per iteration: %.6lf s\n", total_time / (ntimes-1));
    }
}
