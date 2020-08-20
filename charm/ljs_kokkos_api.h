#ifndef LJS_KOKKOS_API_H_
#define LJS_KOKKOS_API_H_

void kokkosInitialize(int num_threads, int teams, int device);
void kokkosFinalize();

void blockNew(void** block, int index, cudaStream_t compute_stream,
    cudaStream_t comm_stream, cudaEvent_t compute_event, cudaEvent_t comm_event);
void blockDelete(void* block);

#endif // LJS_KOKKOS_API_H_
