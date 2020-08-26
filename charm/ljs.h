#ifndef LJS_H_
#define LJS_J_

#include "miniMD.decl.h"
#include "ljs_kokkos.h"

struct InstanceHolder {
  Kokkos::Cuda compute_instance;
  Kokkos::Cuda h2d_instance;
  Kokkos::Cuda d2h_instance;
  Kokkos::Cuda pack_instance;
  Kokkos::Cuda unpack_instance;
};

class KokkosManager : public CBase_KokkosManager {
public:
  cudaStream_t compute_stream;
  cudaStream_t h2d_stream;
  cudaStream_t d2h_stream;
  cudaStream_t pack_stream;
  cudaStream_t unpack_stream;

  InstanceHolder* instances;

  KokkosManager();
  void finalize();
};

#endif
