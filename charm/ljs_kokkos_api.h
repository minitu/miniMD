#ifndef LJS_KOKKOS_API_H_
#define LJS_KOKKOS_API_H_

void kokkosInitialize(int num_threads, int teams, int device);
void kokkosFinalize();

#endif // LJS_KOKKOS_API_H_
