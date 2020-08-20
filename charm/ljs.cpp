#include "miniMD.decl.h"
#include "pup_stl.h"
#include "hapi.h"

#include "types.h"
#include "ljs_kokkos_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_KokkosManager kokkos_proxy;
/* readonly */ CProxy_Block block_proxy;
/* readonly */ int num_chares;

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
    char* input_file = nullptr;

    for (int i = 0; i < m->argc; i++) {
      if ((strcmp(m->argv[i], "-i") == 0) || (strcmp(m->argv[i], "--input_file") == 0)) {
        input_file = m->argv[++i];
        continue;
      }
    }

    int error = 0;
    if (input_file == nullptr) {
      error = input("../inputs/in.lj.miniMD", in_nx, in_ny, in_nz, in_t_request,
          in_rho, in_units, in_forcetype, in_epsilon, in_sigma, in_datafile,
          in_ntimes, in_dt, in_neigh_every, in_force_cut, in_neigh_cut,
          in_thermo_nstat);
    } else {
      error = input(input_file, in_nx, in_ny, in_nz, in_t_request,
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
      printf("# EAM currently requires '--ghost_newton 0'; Changing setting now.\n");
      ghost_newton = 0;
    }

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

    thisProxy.run();
  }
};

class KokkosManager : public CBase_KokkosManager {
public:
  KokkosManager() {
    // Initialize Kokkos
    kokkosInitialize(num_threads, teams, 0);

    contribute(CkCallback(CkReductionTarget(Main, kokkosInitialized), main_proxy));
  }

  void finalize() {
    // Finalize Kokkos
    kokkosFinalize();

    contribute(CkCallback(CkReductionTarget(Main, kokkosFinalized), main_proxy));
  }
};

class Block : public CBase_Block {
  void* block;

  cudaStream_t compute_stream;
  cudaStream_t comm_stream;

  cudaEvent_t compute_event;
  cudaEvent_t comm_event;

public:
  Block() {
    // Create CUDA streams (higher priority for communication stream)
    cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault, 0);
    cudaStreamCreateWithPriority(&comm_stream, cudaStreamDefault, -1);

    // Create CUDA events used to preserve dependencies between streams
    cudaEventCreateWithFlags(&compute_event, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&comm_event, cudaEventDisableTiming);

    blockNew(&block, thisIndex, compute_stream, comm_stream, compute_event,
        comm_event);

    CkCallback* cb = new CkCallback(CkIndex_Block::initDone(), thisProxy[thisIndex]);
    hapiAddCallback(comm_stream, cb);
  }

  void initDone() {
    contribute(CkCallback(CkReductionTarget(Main, blocksCreated), main_proxy));
  }

  ~Block() { blockDelete(block); }
};

#include "miniMD.def.h"
