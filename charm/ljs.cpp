#include "miniMD.decl.h"
#include "pup_stl.h"
#include "hapi.h"
#include "ljs_kokkos_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_KokkosManager kokkos_proxy;

class Main : public CBase_Main {
public:
  Main(CkArgMsg* m) {
    // Pack command line arguments
    std::vector<std::string> args;
    for (int i = 0; i < m->argc; i++) {
      args.push_back(std::string(m->argv[i]));
    }

    // Create KokkosManagers on each process
    kokkos_proxy = CProxy_KokkosManager::ckNew(m->argc, args);
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
  KokkosManager(int argc, std::vector<std::string> args) {
    // Unpack command line arguments
    char* argv[argc];
    for (int i = 0; i < argc; i++) {
      argv[i] = const_cast<char*>(args[i].c_str());
    }

    // Default parameters
    int num_threads = 1;
    int teams = 1;
    int device = 0;

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
