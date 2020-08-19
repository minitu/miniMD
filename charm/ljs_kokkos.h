#ifndef LJS_KOKKOS_H_
#define LJS_KOKKOS_H_

#include "types.h"

#include "Kokkos_Core.hpp"
#include "Kokkos_DualView.hpp"

typedef Kokkos::DefaultExecutionSpace DeviceType;
typedef Kokkos::HostSpace::execution_space HostType;

typedef Kokkos::DualView<MMD_float*[PAD],Kokkos::LayoutRight> x_dual_view_type;
typedef Kokkos::DualView<MMD_float*> float_1d_dual_view_type;
typedef Kokkos::DualView<MMD_float**> float_2d_dual_view_type;
typedef Kokkos::DualView<MMD_int*> int_1d_dual_view_type;
typedef Kokkos::DualView<MMD_int**> int_2d_dual_view_type;
typedef Kokkos::DualView<MMD_int> int_dual_view_type;

typedef Kokkos::View<MMD_float*[PAD],Kokkos::LayoutRight> x_view_type;
typedef Kokkos::View<MMD_float**> float_2d_view_type;
typedef Kokkos::View<MMD_float*> float_1d_view_type;
typedef Kokkos::View<MMD_int*> int_1d_view_type;
typedef Kokkos::View<MMD_int**> int_2d_view_type;
typedef Kokkos::View<MMD_int**,Kokkos::LayoutRight> int_2d_lr_view_type;
typedef Kokkos::View<MMD_int> int_view_type;

typedef Kokkos::View<const MMD_float*[PAD],Kokkos::LayoutRight> x_const_view_type;
typedef Kokkos::View<const MMD_int*> int_1d_const_view_type;
typedef Kokkos::View<const MMD_int**> int_2d_const_view_type;

typedef Kokkos::View<MMD_float*[PAD],Kokkos::LayoutRight,Kokkos::MemoryTraits<Kokkos::Atomic> > x_atomic_view_type;
typedef Kokkos::View<MMD_float**,Kokkos::MemoryTraits<Kokkos::Atomic> > float_2d_atomic_view_type;
typedef Kokkos::View<MMD_float*,Kokkos::MemoryTraits<Kokkos::Atomic> > float_1d_atomic_view_type;
typedef Kokkos::View<MMD_float,Kokkos::MemoryTraits<Kokkos::Atomic> > float_atomic_view_type;
typedef Kokkos::View<MMD_int*,Kokkos::MemoryTraits<Kokkos::Atomic> > int_1d_atomic_view_type;
typedef Kokkos::View<MMD_int**,Kokkos::MemoryTraits<Kokkos::Atomic> > int_2d_atomic_view_type;

typedef Kokkos::View<MMD_float*[PAD],Kokkos::LayoutRight,Kokkos::MemoryTraits<Kokkos::Atomic|Kokkos::Unmanaged> > x_atomic_um_view_type;
typedef Kokkos::View<MMD_float*,Kokkos::MemoryTraits<Kokkos::Atomic|Kokkos::Unmanaged> > float_1d_atomic_um_view_type;

typedef Kokkos::View<const MMD_float*[PAD],Kokkos::LayoutRight,Kokkos::MemoryTraits<Kokkos::RandomAccess> > x_rnd_view_type;
typedef Kokkos::View<const MMD_float**,Kokkos::MemoryTraits<Kokkos::RandomAccess> > float_2d_rnd_view_type;
typedef Kokkos::View<const MMD_float*,Kokkos::MemoryTraits<Kokkos::RandomAccess> > float_1d_rnd_view_type;
typedef Kokkos::View<const MMD_int*,Kokkos::MemoryTraits<Kokkos::RandomAccess> > int_1d_rnd_view_type;
typedef Kokkos::View<const MMD_int**,Kokkos::MemoryTraits<Kokkos::RandomAccess> > int_2d_rnd_view_type;

typedef Kokkos::View<MMD_float*[PAD],Kokkos::LayoutRight,Kokkos::MemoryTraits<Kokkos::Unmanaged>> x_um_view_type;
typedef Kokkos::View<MMD_int*,Kokkos::MemoryTraits<Kokkos::Unmanaged> > int_1d_um_view_type;
typedef Kokkos::View<const MMD_int*,Kokkos::MemoryTraits<Kokkos::Unmanaged> > int_1d_const_um_view_type;

typedef typename x_view_type::HostMirror x_host_view_type;
typedef typename float_1d_view_type::HostMirror float_1d_host_view_type;
typedef typename float_2d_view_type::HostMirror float_2d_host_view_type;
typedef typename int_1d_view_type::HostMirror int_1d_host_view_type;
typedef typename int_2d_view_type::HostMirror int_2d_host_view_type;
typedef typename int_view_type::HostMirror int_host_view_type;

typedef typename Kokkos::DefaultExecutionSpace::scratch_memory_space SharedSpace;
typedef Kokkos::View<float*[3], Kokkos::LayoutLeft, SharedSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > neighbor_pos_shared_type;
typedef Kokkos::View<int*, SharedSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > int_1d_shared_type;

typedef Kokkos::View<int**,Kokkos::LayoutLeft,SharedSpace,Kokkos::MemoryUnmanaged> t_shared_2d_int;
typedef Kokkos::View<float**[3],Kokkos::LayoutLeft,SharedSpace,Kokkos::MemoryUnmanaged> t_shared_pos;

struct eng_virial_type {
  MMD_float eng;
  MMD_float virial;
  KOKKOS_INLINE_FUNCTION
  eng_virial_type() {eng = 0.0; virial = 0.0;}

  KOKKOS_INLINE_FUNCTION
  eng_virial_type& operator += (const eng_virial_type& src) {
    eng+=src.eng;
    virial+=src.virial;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION
  void operator += (const volatile eng_virial_type& src) volatile {
    eng+=src.eng;
    virial+=src.virial;
  }
};

#endif // __LJS_KOKKOS_H_
