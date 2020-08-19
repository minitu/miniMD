#include "miniMD.decl.h"
#include "pup_stl.h"
#include "hapi.h"

#include "ljs.h"
#include "ljs_kokkos_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_KokkosManager kokkos_proxy;

extern int input(In& in, const char* filename);

class Main : public CBase_Main {
public:
  Main(CkArgMsg* m) {
    // Default parameters
    int num_threads = 1;
    int teams = 1;
    int device = 0;
    int num_steps = -1;
    int system_size = -1;
    int nx = -1; int ny = -1; int nz = -1;
    int ntypes = 8;
    int neighbor_size = -1;
    int halfneigh = 1;
    int team_neigh = 0;
    int use_sse = 0;
    int check_safeexchange = 0;
    int do_safeexchange = 0;
    int sort = -1;
    int yaml_output = 0;
    int yaml_screen = 0;
    int ghost_newton = 1;

    // Process input file
    In in;
    in.datafile = nullptr;
    char* input_file = nullptr;

    for (int i = 0; i < m->argc; i++) {
      if ((strcmp(m->argv[i], "-i") == 0) || (strcmp(m->argv[i], "--input_file") == 0)) {
        input_file = m->argv[++i];
        continue;
      }
    }

    int error = 0;
    if (input_file == nullptr) {
      error = input(in, "../inputs/in.lj.miniMD");
    } else {
      error = input(in, input_file);
    }

    if (error) {
      CkPrintf("ERROR: Failed to read input file\n");
      CkExit();
    }

    srand(5413);

    // Process other command line parameters
    for (int i = 0; i < m->argc; i++) {
      if ((strcmp(m->argv[i], "-t") == 0) || (strcmp(m->argv[i], "--num_threads") == 0)) {
        num_threads = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "--teams") == 0)) {
        teams = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "-d") == 0) || (strcmp(m->argv[i], "--device") == 0)) {
        device = atoi(m->argv[++i]);
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
        if (in.datafile == NULL) in.datafile = new char[1000];

        strcpy(in.datafile, m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "-u") == 0) || (strcmp(m->argv[i], "--units") == 0)) {
        in.units = strcmp(m->argv[++i], "metal") == 0 ? 1 : 0;
        continue;
      }

      if ((strcmp(m->argv[i], "-p") == 0) || (strcmp(m->argv[i], "--force") == 0)) {
        in.forcetype = strcmp(m->argv[++i], "eam") == 0 ? FORCEEAM : FORCELJ;
        continue;
      }

      if ((strcmp(m->argv[i], "-gn") == 0) || (strcmp(m->argv[i], "--ghost_newton") == 0)) {
        ghost_newton = atoi(m->argv[++i]);
        continue;
      }
    }

    // Create KokkosManagers on each process
    kokkos_proxy = CProxy_KokkosManager::ckNew(num_threads, teams, device);
  }

  void kokkosInitialized() {
    CkPrintf("Kokkos initialized!\n");
    kokkos_proxy.finalize();
  }

  void kokkosFinalized() {
    CkPrintf("Kokkos finalized!\n");
    CkExit();
  }
};

class KokkosManager : public CBase_KokkosManager {
public:
  KokkosManager(int num_threads, int teams, int device) {
    // Initialize Kokkos
    kokkosInitialize(num_threads, teams, device);

    contribute(CkCallback(CkReductionTarget(Main, kokkosInitialized), main_proxy));
  }

  void finalize() {
    // Finalize Kokkos
    kokkosFinalize();

    contribute(CkCallback(CkReductionTarget(Main, kokkosFinalized), main_proxy));
  }
};

#include "miniMD.def.h"
