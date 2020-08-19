#include "ljs.h"
#include "ljs_kokkos.h"

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
