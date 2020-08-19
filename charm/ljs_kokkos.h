#ifndef __LJS_KOKKOS_H_
#define __LJS_KOKKOS_H_

void kokkosInitialize(int num_threads, int teams, int device);
void kokkosFinalize();

#endif // __LJS_KOKKOS_H_
