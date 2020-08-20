#include "ljs_kokkos.h"
#include "ljs_kokkos_api.h"
#include "atom.h"
#include "neighbor.h"
#include "integrate.h"
#include "thermo.h"
#include "comm.h"
#include "force.h"
#include "force_lj.h"

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
/* readonly */ extern int in_nx;
/* readonly */ extern int in_ny;
/* readonly */ extern int in_nz;
/* readonly */ extern MMD_float in_t_request;
/* readonly */ extern MMD_float in_rho;
/* readonly */ extern int in_units;
/* readonly */ extern ForceStyle in_forcetype;
/* readonly */ extern MMD_float in_epsilon;
/* readonly */ extern MMD_float in_sigma;
/* readonly */ extern std::string in_datafile;
/* readonly */ extern int in_ntimes;
/* readonly */ extern MMD_float in_dt;
/* readonly */ extern int in_neigh_every;
/* readonly */ extern MMD_float in_force_cut;
/* readonly */ extern MMD_float in_neigh_cut;
/* readonly */ extern int in_thermo_nstat;

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

struct BlockKokkos {
  Atom atom;
  Neighbor neighbor;
  Integrate integrate;
  Thermo thermo;
  Comm comm;
  Force* force;

  BlockKokkos() : atom(ntypes), neighbor(ntypes), integrate(), thermo(), comm(), force(NULL) {
    if (in_forcetype == FORCEEAM) {
      force = (Force*) new ForceEAM(ntypes);
    }
  }
};

void blockNew(void** block) {
  *block = (void*)new BlockKokkos;
}

void blockDelete(void* block) {
  delete (BlockKokkos*)block;
}
