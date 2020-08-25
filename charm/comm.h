/* ----------------------------------------------------------------------
   miniMD is a simple, parallel molecular dynamics (MD) code.   miniMD is
   an MD microapplication in the Mantevo project at Sandia National
   Laboratories ( http://www.mantevo.org ). The primary
   authors of miniMD are Steve Plimpton (sjplimp@sandia.gov) , Paul Crozier
   (pscrozi@sandia.gov) and Christian Trott (crtrott@sandia.gov).

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This library is free software; you
   can redistribute it and/or modify it under the terms of the GNU Lesser
   General Public License as published by the Free Software Foundation;
   either version 3 of the License, or (at your option) any later
   version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this software; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
   USA.  See also: http://www.gnu.org/licenses/lgpl.txt .

   For questions, contact Paul S. Crozier (pscrozi@sandia.gov) or
   Christian Trott (crtrott@sandia.gov).

   Please read the accompanying README and LICENSE files.
---------------------------------------------------------------------- */

#ifndef COMM_H
#define COMM_H

#include "block.decl.h"
#include "atom.h"

class Comm : public CBase_Comm
{
  public:

    struct TagExchangeSendlist {};
    struct TagExchangePack {};
    struct TagExchangeCountRecv {};
    struct TagExchangeUnpack {};
    struct TagBorderSendlist {};
    struct TagBorderPack {};
    struct TagBorderUnpack {};
    typedef int value_type;

    Kokkos::Cuda compute_instance;
    Kokkos::Cuda comm_instance;

    cudaEvent_t compute_event;
    cudaEvent_t comm_event;

    KOKKOS_INLINE_FUNCTION
    void operator() (TagExchangeSendlist, const int&  ) const;
    KOKKOS_INLINE_FUNCTION
    void operator() (TagExchangePack, const int&  ) const;
    KOKKOS_INLINE_FUNCTION
    void operator() (TagExchangeCountRecv, const int& , int& ) const;
    KOKKOS_INLINE_FUNCTION
    void operator() (TagExchangeUnpack, const int&  ) const;
    KOKKOS_INLINE_FUNCTION
    void operator() (TagBorderSendlist, const int&  ) const;
    KOKKOS_INLINE_FUNCTION
    void operator() (TagBorderPack, const int&  ) const;
    KOKKOS_INLINE_FUNCTION
    void operator() (TagBorderUnpack, const int&  ) const;

    Comm();
    ~Comm();
    void init();
    int setup(MMD_float, Atom &);
    void communicate(Atom &, bool);
    void reverse_communicate(Atom &, bool);
    void exchange(Atom &, bool);
    void borders(Atom &, bool);
    //void send(int, int, CkCallback cb);
    void growsend(int);
    void growrecv(int);
    void growlist(int, int);
    void suspend(Kokkos::Cuda);

  public:
    void* block;
    int iter;

    int index;                                      // my chare index
    int nswap;                                   // # of swaps to perform
    int_1d_host_view_type pbc_any;                    // whether any PBC on this swap
    int_1d_host_view_type pbc_flagx;                  // PBC correction in x for this swap
    int_1d_host_view_type pbc_flagy;                  // same in y
    int_1d_host_view_type pbc_flagz;                  // same in z
    int_1d_host_view_type sendnum, recvnum;           // # of atoms to send/recv in each swap
    int_1d_host_view_type comm_send_size;             // # of values to send in each comm
    int_1d_host_view_type comm_recv_size;             // # of values to recv in each comm
    int_1d_host_view_type reverse_send_size;          // # of values to send in each reverse
    int_1d_host_view_type reverse_recv_size;          // # of values to recv in each reverse
    int_1d_host_view_type sendchare, recvchare;       // chare to send/recv with at each swap

    int_1d_host_view_type firstrecv;                  // where to put 1st recv atom in each swap
    int_2d_lr_view_type sendlist;                   // list of atoms to send in each swap
    int_1d_host_view_type maxsendlist;

    int_1d_view_type exc_sendflag;
    int_1d_view_type exc_sendlist;
    int_1d_view_type exc_copylist;
    int_1d_host_view_type h_exc_sendflag;
    int_1d_host_view_type h_exc_sendlist;
    int_1d_host_view_type h_exc_copylist;
    int_1d_dual_view_type count;
    bool h_exc_alloc;

    float_1d_view_type buf_send;                 // send buffer for all comm
    float_1d_view_type buf_recv;                 // recv buffer for all comm
    float_1d_view_type buf;
    float_1d_host_view_type h_buf_send;
    float_1d_host_view_type h_buf_recv;
    bool h_buf_alloc;
    int maxsend;
    int maxrecv;

    int chareneigh[3][2];              // my 6 chare neighbors
    int charegrid[3];                  // # of chares in each dim
    int need[3];                      // how many chares away needed in each dim
    float_1d_host_view_type slablo, slabhi;          // bounds of slabs to send to other procs

    int check_safeexchange;           // if sets give warnings if an atom moves further than subdomain size
    int do_safeexchange;		    // exchange atoms with all subdomains within neighbor cutoff

    int copy_size;
    int_1d_view_type send_flag;
    int maxnlocal;
    int nrecv_atoms;

    // Used for Charm++ communication
    int nsend, nrecv, nrecv1, nrecv2, nlocal;
    void *send1, *send2, *recv1, *recv2;
    size_t send1_size, send2_size;
    int send1_chare, send2_chare, recv1_chare, recv2_chare;
    CkCallbackResumeThread* resume_cb;

  private:
    Atom atom;
    int idim,n,iswap;
    MMD_float lo, hi, value;
    x_view_type x;
    int pbc_flags[4];
    int_1d_atomic_view_type send_count;
};

#endif
