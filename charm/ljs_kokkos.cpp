#include "ljs_kokkos.h"
#include "ljs_kokkos_api.h"
#include "atom.h"
/*
#include "neighbor.h"
#include "integrate.h"
#include "thermo.h"
#include "comm.h"
#include "timer.h"
#include "force.h"
#include "force_lj.h"
*/

/* readonly */ extern int num_threads;
/* readonly */ extern int teams;
/* readonly */ extern int num_steps;
/* readonly */ extern int system_size;
/* readonly */ extern int nx;
/* readonly */ extern int ny;
/* readonly */ extern int nz;
/* readonly */ extern int ntypes;
/* readonly */ extern int neighbor_size;
/* readonly */ extern int halfneigh;
/* readonly */ extern int team_neigh;
/* readonly */ extern int use_sse;
/* readonly */ extern int check_safeexchange;
/* readonly */ extern int do_safeexchange;
/* readonly */ extern int sort;
/* readonly */ extern int yaml_output;
/* readonly */ extern int yaml_screen;
/* readonly */ extern int ghost_newton;

void kokkosInitialize(int num_threads, int teams, int device) {
  Kokkos::InitArguments args_kokkos;
  args_kokkos.num_threads = num_threads;
  args_kokkos.num_numa = teams;
  args_kokkos.device_id = device;
  Kokkos::initialize(args_kokkos);
}

void kokkosFinalize() {
  Kokkos::finalize();
}

void blockKokkos() {
  Atom atom(ntypes);
  /*
  Neighbor neighbor(ntypes);
  Integrate integrate;
  Thermo thermo;
  Comm comm;
  Timer timer;

  Force* force;
  */
}
