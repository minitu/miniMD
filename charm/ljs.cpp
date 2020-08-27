#include "miniMD.decl.h"
#include "pup_stl.h"
#include "hapi.h"

#include "ljs_kokkos.h"
#include "ljs.h"

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_KokkosManager kokkos_proxy;
/* readonly */ CProxy_Block block_proxy;
/* readonly */ CProxy_Comm comm_proxy;
/* readonly */ int num_chares;

/* readonly */ std::string input_file;
/* readonly */ int num_threads;
/* readonly */ int teams;
/* readonly */ int num_steps;
/* readonly */ int system_size;
/* readonly */ int nx;
/* readonly */ int ny;
/* readonly */ int nz;
/* readonly */ int ntypes;
/* readonly */ int neighbor_size;
/* readonly */ int halfneigh;
/* readonly */ int team_neigh;
/* readonly */ int use_sse;
/* readonly */ int check_safeexchange;
/* readonly */ int do_safeexchange;
/* readonly */ int sort;
/* readonly */ int yaml_output;
/* readonly */ int yaml_screen;
/* readonly */ int ghost_newton;
/* readonly */ int in_nx;
/* readonly */ int in_ny;
/* readonly */ int in_nz;
/* readonly */ MMD_float in_t_request;
/* readonly */ MMD_float in_rho;
/* readonly */ int in_units;
/* readonly */ ForceStyle in_forcetype;
/* readonly */ MMD_float in_epsilon;
/* readonly */ MMD_float in_sigma;
/* readonly */ std::string in_datafile;
/* readonly */ int in_ntimes;
/* readonly */ MMD_float in_dt;
/* readonly */ int in_neigh_every;
/* readonly */ MMD_float in_force_cut;
/* readonly */ MMD_float in_neigh_cut;
/* readonly */ int in_thermo_nstat;

extern int input(const char* filename, int& in_nx, int& in_ny, int& in_nz,
    MMD_float& in_t_request, MMD_float& in_rho, int& in_units,
    ForceStyle& in_forcetype, MMD_float& in_epsilon, MMD_float& in_sigma,
    std::string& in_datafile, int& in_ntimes, MMD_float& in_dt,
    int& in_neigh_every, MMD_float& in_force_cut, MMD_float& in_neigh_cut,
    int& in_thermo_nstat);

class Main : public CBase_Main {
  Main_SDAG_CODE

public:
  Main(CkArgMsg* m) {
    // Default parameters
    num_chares = 4;
    num_threads = 1;
    teams = 1;
    num_steps = -1;
    system_size = -1;
    nx = -1; int ny = -1; int nz = -1;
    ntypes = 8;
    neighbor_size = -1;
    halfneigh = 1;
    team_neigh = 0;
    use_sse = 0;
    check_safeexchange = 0;
    do_safeexchange = 0;
    sort = -1;
    yaml_output = 0;
    yaml_screen = 0;
    ghost_newton = 1;

    // Process input file
    for (int i = 0; i < m->argc; i++) {
      if ((strcmp(m->argv[i], "-i") == 0) || (strcmp(m->argv[i], "--input_file") == 0)) {
        input_file = std::string(m->argv[++i]);
        continue;
      }
    }

    int error = 0;
    if (input_file.empty()) {
      error = input("../inputs/in.lj.miniMD", in_nx, in_ny, in_nz, in_t_request,
          in_rho, in_units, in_forcetype, in_epsilon, in_sigma, in_datafile,
          in_ntimes, in_dt, in_neigh_every, in_force_cut, in_neigh_cut,
          in_thermo_nstat);
    } else {
      error = input(input_file.c_str(), in_nx, in_ny, in_nz, in_t_request,
          in_rho, in_units, in_forcetype, in_epsilon, in_sigma, in_datafile,
          in_ntimes, in_dt, in_neigh_every, in_force_cut, in_neigh_cut,
          in_thermo_nstat);
    }

    if (error) {
      CkPrintf("ERROR: Failed to read input file\n");
      CkExit();
    }

    srand(5413);

    // Process other command line parameters
    for (int i = 0; i < m->argc; i++) {
      if ((strcmp(m->argv[i], "-c") == 0) || (strcmp(m->argv[i], "--num_chares") == 0)) {
        num_chares = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "-t") == 0) || (strcmp(m->argv[i], "--num_threads") == 0)) {
        num_threads = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "--teams") == 0)) {
        teams = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "-n") == 0) || (strcmp(m->argv[i], "--nsteps") == 0))  {
        num_steps = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "-s") == 0) || (strcmp(m->argv[i], "--size") == 0)) {
        system_size = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "-nx") == 0)) {
        nx = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "-ny") == 0)) {
        ny = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "-nz") == 0)) {
        nz = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "--ntypes") == 0)) {
        ntypes = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "-b") == 0) || (strcmp(m->argv[i], "--neigh_bins") == 0))  {
        neighbor_size = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "--half_neigh") == 0))  {
        halfneigh = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "--team_neigh") == 0))  {
        team_neigh = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "-sse") == 0))  {
        use_sse = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "--check_exchange") == 0))  {
        check_safeexchange = 1;
        continue;
      }

      if ((strcmp(m->argv[i], "--safe_exchange") == 0)) {
        do_safeexchange = 1;
        continue;
      }

      if ((strcmp(m->argv[i], "--sort") == 0))  {
        sort = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "-o") == 0) || (strcmp(m->argv[i], "--yaml_output") == 0))  {
        yaml_output = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "--yaml_screen") == 0))  {
        yaml_screen = 1;
        continue;
      }

      if ((strcmp(m->argv[i], "-f") == 0) || (strcmp(m->argv[i], "--data_file") == 0)) {
        in_datafile = std::string(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "-u") == 0) || (strcmp(m->argv[i], "--units") == 0)) {
        in_units = strcmp(m->argv[++i], "metal") == 0 ? 1 : 0;
        continue;
      }

      if ((strcmp(m->argv[i], "-p") == 0) || (strcmp(m->argv[i], "--force") == 0)) {
        in_forcetype = strcmp(m->argv[++i], "eam") == 0 ? FORCEEAM : FORCELJ;
        continue;
      }

      if ((strcmp(m->argv[i], "-gn") == 0) || (strcmp(m->argv[i], "--ghost_newton") == 0)) {
        ghost_newton = atoi(m->argv[++i]);
        continue;
      }
    }

    if (in_forcetype == FORCEEAM && ghost_newton == 1) {
      CkPrintf("# EAM currently requires '--ghost_newton 0'; Changing setting now.\n");
      ghost_newton = 0;
    }

    // Done here since global variables can only be modified here
    if (num_steps > 0) in_ntimes = num_steps;
    if (system_size > 0) in_nx = in_ny = in_nz = system_size;

    if (nx > 0) {
      in_nx = nx;
      if (ny > 0) in_ny = ny;
      else if (system_size < 0) in_ny = nx;

      if (nz > 0) in_nz = nz;
      else if (system_size < 0) in_nz = nx;
    }

    // Create KokkosManagers on each process
    kokkos_proxy = CProxy_KokkosManager::ckNew();

    // Create block and comm chare arrays
    // Their constructors shouldn't call Kokkos functions
    block_proxy = CProxy_Block::ckNew(num_chares);
    CkArrayOptions opts(num_chares);
    opts.bindTo(block_proxy);
    comm_proxy = CProxy_Comm::ckNew(opts);

    thisProxy.run();
  }
};

KokkosManager::KokkosManager() {
  // Initialize Kokkos
  Kokkos::InitArguments args_kokkos;
  args_kokkos.num_threads = num_threads;
  args_kokkos.num_numa = teams;
  args_kokkos.device_id = 0;
  Kokkos::initialize(args_kokkos);

  // Create per-GPU streams (only works with 1 process per GPU)
  cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault, 0);
  cudaStreamCreateWithPriority(&h2d_stream, cudaStreamDefault, -1);
  cudaStreamCreateWithPriority(&d2h_stream, cudaStreamDefault, -1);
  cudaStreamCreateWithPriority(&pack_stream, cudaStreamDefault, -1);
  cudaStreamCreateWithPriority(&unpack_stream, cudaStreamDefault, -1);

  // Create CUDA execution instances using streams
  instances = new InstanceHolder;
  instances->compute_instance = Kokkos::Cuda(compute_stream);
  instances->h2d_instance = Kokkos::Cuda(h2d_stream);
  instances->d2h_instance = Kokkos::Cuda(d2h_stream);
  instances->pack_instance = Kokkos::Cuda(pack_stream);
  instances->unpack_instance = Kokkos::Cuda(unpack_stream);

  contribute(CkCallback(CkReductionTarget(Main, kokkosInitialized), main_proxy));
}

void KokkosManager::finalize() {
  // Destroy instance holder
  delete instances;

  // Destroy streams
  cudaStreamDestroy(compute_stream);
  cudaStreamDestroy(h2d_stream);
  cudaStreamDestroy(d2h_stream);
  cudaStreamDestroy(pack_stream);
  cudaStreamDestroy(unpack_stream);

  // Finalize Kokkos
  Kokkos::finalize();

  contribute(CkCallback(CkReductionTarget(Main, kokkosFinalized), main_proxy));
}

#include "miniMD.def.h"
