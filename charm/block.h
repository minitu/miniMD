#ifndef BLOCK_H_
#define BLOCK_H_

//#include "block.decl.h"
#include "hapi.h"

#include "ljs_kokkos.h"
#include "atom.h"
#include "neighbor.h"
#include "integrate.h"
#include "thermo.h"
#include "comm.h"
#include "force.h"

// Virtualization of a process in the MPI version
class Block : public CBase_Block {
  Block_SDAG_CODE

  Atom atom;
  Neighbor neighbor;
  Integrate integrate;
  Thermo thermo;
  Comm* comm;
  Force* force;

  int iter;

  cudaStream_t compute_stream;
  cudaStream_t comm_stream;

  cudaEvent_t compute_event;
  cudaEvent_t comm_event;

  Kokkos::Cuda compute_instance;
  Kokkos::Cuda comm_instance;

  double vtot[3];

public:
  Block();

  void saveBoundArray();
  void init();
  void contCreateVelocity(double vxtot, double vytot, double vztot);
  void printConfig();

  ~Block() {}
};

#endif // BLOCK_H_
