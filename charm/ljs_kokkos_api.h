#ifndef LJS_KOKKOS_API_H_
#define LJS_KOKKOS_API_H_

void kokkosInitialize(int num_threads, int teams, int device);
void kokkosFinalize();

void blockVelocity(void* block, double vxtot, double vytot, double vztot);
void blockNew(void** block, int index, cudaStream_t compute_stream,
    cudaStream_t comm_stream, cudaEvent_t compute_event, cudaEvent_t comm_event,
    double& vxtot, double& vytot, double& vztot);
void blockDelete(void* block);

#endif // LJS_KOKKOS_API_H_
