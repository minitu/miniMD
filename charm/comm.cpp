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

#include "stdio.h"
#include "stdlib.h"
#include "comm.h"
#include "hapi_nvtx.h"

#define BUFFACTOR 1.5
#define BUFMIN 1000
#define BUFEXTRA 100
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define IDX(x,y,z) ((charegrid[0] * charegrid[1] * (z)) + charegrid[0] * (y) + (x))

#include "miniMD.decl.h"
#include "block.decl.h"
#include "block.h"

/* readonly */ extern CProxy_Main main_proxy;
/* readonly */ extern CProxy_Block block_proxy;
/* readonly */ extern CProxy_Comm comm_proxy;
/* readonly */ extern int num_chares;

Comm::Comm()
{
  index = thisIndex;
  maxsend = BUFMIN;
  buf_send = float_1d_view_type("Comm::buf_send",maxsend + BUFMIN);
  maxrecv = BUFMIN;
  buf_recv = float_1d_view_type("Comm::buf_recv",maxrecv);
  check_safeexchange = 0;
  do_safeexchange = 0;
  maxnlocal = 0;
  count = Kokkos::DualView<int*>("comm::count",1);
  h_exc_alloc = false;
  h_buf_alloc = false;

  // Save pointer to Block bound array element
  block = (void*)block_proxy(thisIndex).ckLocal();
  iter = 0;
}

Comm::~Comm() {}

/* setup spatial-decomposition communication patterns */

int Comm::setup(MMD_float cutneigh, Atom &atom)
{
  int i;
  MMD_float prd[3];
  int myloc[3];
  MMD_float lo, hi;
  int ineed, idim, nbox;

  prd[0] = atom.box.xprd;
  prd[1] = atom.box.yprd;
  prd[2] = atom.box.zprd;

  /* setup 3-d grid of chares */

  MMD_float area[3];

  area[0] = prd[0] * prd[1];
  area[1] = prd[0] * prd[2];
  area[2] = prd[1] * prd[2];

  MMD_float bestsurf = 2.0 * (area[0] + area[1] + area[2]);

  // loop thru all possible factorizations of num_chares
  // surf = surface area of a chare sub-domain
  // for 2d, insure ipz = 1

  int ipx, ipy, ipz, nremain;
  MMD_float surf;

  ipx = 1;

  while (ipx <= num_chares) {
    if (num_chares % ipx == 0) {
      nremain = num_chares / ipx;
      ipy = 1;

      while (ipy <= nremain) {
        if (nremain % ipy == 0) {
          ipz = nremain / ipy;
          surf = area[0] / ipx / ipy + area[1] / ipx / ipz + area[2] / ipy / ipz;

          if (surf < bestsurf) {
            bestsurf = surf;
            charegrid[0] = ipx;
            charegrid[1] = ipy;
            charegrid[2] = ipz;
          }
        }

        ipy++;
      }
    }

    ipx++;
  }

  if (index == 0) {
    printf("Chare grid: %d x %d x %d\n", charegrid[0], charegrid[1], charegrid[2]);
  }

  if (charegrid[0]*charegrid[1]*charegrid[2] != num_chares) {
    if (index == 0) printf("ERROR: Bad grid of chares\n");

    return 1;
  }

  /* determine where I am and my neighboring procs in 3d grid of chares */
  myloc[0] = index % charegrid[0];
  myloc[1] = (index / charegrid[0]) % charegrid[1];
  myloc[2] = index / (charegrid[0] * charegrid[1]);

  if (myloc[0] == 0)              chareneigh[0][0] = IDX(charegrid[0]-1,myloc[1],myloc[2]);
  else                            chareneigh[0][0] = IDX(myloc[0]-1,myloc[1],myloc[2]);
  if (myloc[0] == charegrid[0]-1) chareneigh[0][1] = IDX(0,myloc[1],myloc[2]);
  else                            chareneigh[0][1] = IDX(myloc[0]+1,myloc[1],myloc[2]);
  if (myloc[1] == 0)              chareneigh[1][0] = IDX(myloc[0],charegrid[1]-1,myloc[2]);
  else                            chareneigh[1][0] = IDX(myloc[0],myloc[1]-1,myloc[2]);
  if (myloc[1] == charegrid[1]-1) chareneigh[1][1] = IDX(myloc[0],0,myloc[2]);
  else                            chareneigh[1][1] = IDX(myloc[0],myloc[1]+1,myloc[2]);
  if (myloc[2] == 0)              chareneigh[2][0] = IDX(myloc[0],myloc[1],charegrid[2]-1);
  else                            chareneigh[2][0] = IDX(myloc[0],myloc[1],myloc[2]-1);
  if (myloc[2] == charegrid[2]-1) chareneigh[2][1] = IDX(myloc[0],myloc[1],0);
  else                            chareneigh[2][1] = IDX(myloc[0],myloc[1],myloc[2]+1);

  /*
  CkPrintf("Chare %d (%d, %d, %d), neighbors: %d, %d, %d, %d, %d, %d\n",
      index, myloc[0], myloc[1], myloc[2], chareneigh[0][0], chareneigh[0][1],
      chareneigh[1][0], chareneigh[1][1], chareneigh[2][0], chareneigh[2][1]);
      */

  /* lo/hi = my local box bounds */

  atom.box.xlo = myloc[0] * prd[0] / charegrid[0];
  atom.box.xhi = (myloc[0] + 1) * prd[0] / charegrid[0];
  atom.box.ylo = myloc[1] * prd[1] / charegrid[1];
  atom.box.yhi = (myloc[1] + 1) * prd[1] / charegrid[1];
  atom.box.zlo = myloc[2] * prd[2] / charegrid[2];
  atom.box.zhi = (myloc[2] + 1) * prd[2] / charegrid[2];

  /* need = # of boxes I need atoms from in each dimension */

  need[0] = static_cast<int>(cutneigh * charegrid[0] / prd[0] + 1);
  need[1] = static_cast<int>(cutneigh * charegrid[1] / prd[1] + 1);
  need[2] = static_cast<int>(cutneigh * charegrid[2] / prd[2] + 1);

  /* alloc comm memory */

  int maxswap = 2 * (need[0] + need[1] + need[2]);
  maxswap_static = maxswap;

  slablo = float_1d_host_view_type("Comm::slablo",maxswap);
  slabhi = float_1d_host_view_type("Comm::slabhi",maxswap);
  pbc_any = int_1d_host_view_type("Comm::pbc_any",maxswap);
  pbc_flagx = int_1d_host_view_type("Comm::pbc_flagx",maxswap);
  pbc_flagy = int_1d_host_view_type("Comm::pbc_flagy",maxswap);
  pbc_flagz = int_1d_host_view_type("Comm::pbc_flagz",maxswap);
  sendchare = int_1d_host_view_type("Comm::sendchare",maxswap);
  recvchare = int_1d_host_view_type("Comm::recvchare",maxswap);
  sendnum = int_1d_host_view_type("Comm::sendnum",maxswap);
  recvnum = int_1d_host_view_type("Comm::recvnum",maxswap);
  comm_send_size = int_1d_host_view_type("Comm::comm_send_size",maxswap);
  comm_recv_size = int_1d_host_view_type("Comm::comm_recv_size",maxswap);
  reverse_send_size = int_1d_host_view_type("Comm::reverse_send_size",maxswap);
  reverse_recv_size = int_1d_host_view_type("Comm::reverse_recv_size",maxswap);
  firstrecv = int_1d_host_view_type("Comm::firstrecv",maxswap);
  maxsendlist = int_1d_host_view_type("Comm::maxsendlist",maxswap);

  // XXX: No equivalents to sendproc_exc and recvproc_exc as they were not used
  // in the original code

  for(i = 0; i < maxswap; i++) maxsendlist[i] = BUFMIN;

  sendlist = int_2d_lr_view_type("Comm::sendlist",maxswap,BUFMIN);

  /* setup 4 parameters for each exchange: (spart,rpart,slablo,slabhi)
     sendchare(nswap) = chare to send to at each swap
     recvchare(nswap) = chare to recv from at each swap
     slablo/slabhi(nswap) = slab boundaries (in correct dimension) of atoms
                            to send at each swap
     1st part of if statement is sending to the west/south/down
     2nd part of if statement is sending to the east/north/up
     nbox = atoms I send originated in this box */

  /* set commflag if atoms are being exchanged across a box boundary
     commflag(idim,nswap) =  0 -> not across a boundary
                          =  1 -> add box-length to position when sending
                          = -1 -> subtract box-length from pos when sending */

  nswap = 0;

  for(idim = 0; idim < 3; idim++) {
    for(ineed = 0; ineed < 2 * need[idim]; ineed++) {
      pbc_any[nswap] = 0;
      pbc_flagx[nswap] = 0;
      pbc_flagy[nswap] = 0;
      pbc_flagz[nswap] = 0;

      if(ineed % 2 == 0) {
        sendchare[nswap] = chareneigh[idim][0];
        recvchare[nswap] = chareneigh[idim][1];
        nbox = myloc[idim] + ineed / 2;
        lo = nbox * prd[idim] / charegrid[idim];

        if(idim == 0) hi = atom.box.xlo + cutneigh;

        if(idim == 1) hi = atom.box.ylo + cutneigh;

        if(idim == 2) hi = atom.box.zlo + cutneigh;

        hi = MIN(hi, (nbox + 1) * prd[idim] / charegrid[idim]);

        if(myloc[idim] == 0) {
          pbc_any[nswap] = 1;

          if(idim == 0) pbc_flagx[nswap] = 1;

          if(idim == 1) pbc_flagy[nswap] = 1;

          if(idim == 2) pbc_flagz[nswap] = 1;
        }
      } else {
        sendchare[nswap] = chareneigh[idim][1];
        recvchare[nswap] = chareneigh[idim][0];
        nbox = myloc[idim] - ineed / 2;
        hi = (nbox + 1) * prd[idim] / charegrid[idim];

        if(idim == 0) lo = atom.box.xhi - cutneigh;

        if(idim == 1) lo = atom.box.yhi - cutneigh;

        if(idim == 2) lo = atom.box.zhi - cutneigh;

        lo = MAX(lo, nbox * prd[idim] / charegrid[idim]);

        if(myloc[idim] == charegrid[idim] - 1) {
          pbc_any[nswap] = 1;

          if(idim == 0) pbc_flagx[nswap] = -1;

          if(idim == 1) pbc_flagy[nswap] = -1;

          if(idim == 2) pbc_flagz[nswap] = -1;
        }
      }

      slablo[nswap] = lo;
      slabhi[nswap] = hi;
      nswap++;
    }
  }

  return 0;
}

/* communication of atom info every timestep */

void Comm::communicate(Atom &atom, bool preprocess)
{
  /*
  std::ostringstream os;
  os << "Comm::communicate " << index;
  NVTXTracer(os.str(), NVTXColor::PeterRiver);
  */
  Kokkos::Profiling::pushRegion("Comm::communicate");

  // Create host mirrors for integrate loop
  if (!preprocess && !h_buf_alloc) {
    h_buf_alloc = true;
    buf_comms_send = new float_1d_view_type[nswap];
    buf_comms_recv = new float_1d_view_type[nswap];
    h_buf_comms_send = new float_1d_host_view_type[nswap];
    h_buf_comms_recv = new float_1d_host_view_type[nswap];
    for (int i = 0; i < nswap; i++) {
      buf_comms_send[i] = float_1d_view_type("Comm::buf_comms_send", maxsend + BUFEXTRA);
      buf_comms_recv[i] = float_1d_view_type("Comm::buf_comms_recv", maxrecv);
      h_buf_comms_send[i] = Kokkos::create_mirror_view(Kokkos::CudaHostPinnedSpace(), buf_comms_send[i]);
      h_buf_comms_recv[i] = Kokkos::create_mirror_view(Kokkos::CudaHostPinnedSpace(), buf_comms_recv[i]);
    }
  }

  int iswap;
  int pbc_flags[4];

  if (preprocess) {
    // Send and recv one buffer at a time
    for(iswap = 0; iswap < nswap; iswap++) {
      pbc_flags[0] = pbc_any[iswap];
      pbc_flags[1] = pbc_flagx[iswap];
      pbc_flags[2] = pbc_flagy[iswap];
      pbc_flags[3] = pbc_flagz[iswap];

      int_1d_view_type list = Kokkos::subview(sendlist,iswap,Kokkos::ALL());

      // Pack buffer
      if (sendchare[iswap] != index) {
        atom.pack_comm(sendnum[iswap], list, buf_send, pbc_flags);
        Kokkos::fence();

        // Exchange with another proc
        // If self, set recv buffer to send buffer

        // Move data on device to host for communication
        h_buf_send = Kokkos::create_mirror_view(buf_send);
        h_buf_recv = Kokkos::create_mirror_view(buf_recv);
        Kokkos::deep_copy(h_buf_send, buf_send);

        // Send and suspend
        send1 = h_buf_send.data();
        send1_size = comm_send_size[iswap] * sizeof(MMD_float);
        send1_chare = sendchare[iswap];
        recv1 = h_buf_recv.data();
        block_proxy[thisIndex].comms(iswap, CkCallbackResumeThread());

        // Move received data to device
        Kokkos::deep_copy(buf_recv, h_buf_recv);

        // Unpack received data
        buf = buf_recv;
        atom.unpack_comm(recvnum[iswap], firstrecv[iswap], buf);
      } else {
        // No need to synchronize for self packing
        atom.pack_comm_self(sendnum[iswap], list, firstrecv[iswap], pbc_flags);
      }

      Kokkos::fence();
    }
  } else {
#if !defined PACK_UNPACK_COMPUTE
    // Enforce compute -> pack dependency
    cudaEvent_t dep_event_1;
    hapiCheck(cudaEventCreateWithFlags(&dep_event_1, cudaEventDisableTiming));
    hapiCheck(cudaEventRecord(dep_event_1, compute_instance.cuda_stream()));
    hapiCheck(cudaStreamWaitEvent(pack_instance.cuda_stream(), dep_event_1, 0));
#endif

    // Pack and move buffers to host
    for (iswap = 0; iswap < nswap; iswap++) {
      pbc_flags[0] = pbc_any[iswap];
      pbc_flags[1] = pbc_flagx[iswap];
      pbc_flags[2] = pbc_flagy[iswap];
      pbc_flags[3] = pbc_flagz[iswap];

      int_1d_view_type list = Kokkos::subview(sendlist,iswap,Kokkos::ALL());

      if (sendchare[iswap] != index) {
        // Invoke packing kernel
        atom.pack_comm(sendnum[iswap], list, buf_comms_send[iswap], pbc_flags);

#ifdef PACK_UNPACK_COMPUTE
        // Enforce compute -> d2h dependency
        cudaEvent_t dep_event;
        hapiCheck(cudaEventCreateWithFlags(&dep_event, cudaEventDisableTiming));
        hapiCheck(cudaEventRecord(dep_event, compute_instance.cuda_stream()));
        hapiCheck(cudaStreamWaitEvent(d2h_instance.cuda_stream(), dep_event, 0));
#else
        // Enforce pack -> d2h dependency
        cudaEvent_t dep_event;
        hapiCheck(cudaEventCreateWithFlags(&dep_event, cudaEventDisableTiming));
        hapiCheck(cudaEventRecord(dep_event, pack_instance.cuda_stream()));
        hapiCheck(cudaStreamWaitEvent(d2h_instance.cuda_stream(), dep_event, 0));
#endif

        // Invoke D2H transfer
        Kokkos::deep_copy(d2h_instance, h_buf_comms_send[iswap], buf_comms_send[iswap]);
      } else {
        atom.pack_comm_self(sendnum[iswap], list, firstrecv[iswap], pbc_flags);
      }
    }

#ifdef CUDA_SYNC
    d2h_instance.fence();
#else
    suspend(d2h_instance);
#endif

    // All buffers copied to host, send to neighbors
    // After receiving, move buffers to device and unpack
    atom_p = &atom;
    block_proxy[thisIndex].comm_all(CkCallbackResumeThread());

#if !defined PACK_UNPACK_COMPUTE
    // Enforce unpack -> compute dependency
    cudaEvent_t dep_event;
    hapiCheck(cudaEventCreateWithFlags(&dep_event, cudaEventDisableTiming));
    hapiCheck(cudaEventRecord(dep_event, unpack_instance.cuda_stream()));
    hapiCheck(cudaStreamWaitEvent(compute_instance.cuda_stream(), dep_event, 0));
#endif
  }

  Kokkos::Profiling::popRegion();
}

/* reverse communication of atom info every timestep */

void Comm::reverse_communicate(Atom &atom, bool preprocess)
{
  /*
  std::ostringstream os;
  os << "Comm::reverse_communicate " << index;
  NVTXTracer(os.str(), NVTXColor::PeterRiver);
  */
  Kokkos::Profiling::pushRegion("Comm::reverse_communicate");

  // Create host mirrors for integrate loop
  if (!preprocess && !h_buf_alloc) {
    h_buf_alloc = true;
    buf_comms_send = new float_1d_view_type[nswap];
    buf_comms_recv = new float_1d_view_type[nswap];
    h_buf_comms_send = new float_1d_host_view_type[nswap];
    h_buf_comms_recv = new float_1d_host_view_type[nswap];
    for (int i = 0; i < nswap; i++) {
      buf_comms_send[i] = float_1d_view_type("Comm::buf_comms_send", maxsend + BUFEXTRA);
      buf_comms_recv[i] = float_1d_view_type("Comm::buf_comms_recv", maxrecv);
      h_buf_comms_send[i] = Kokkos::create_mirror_view(Kokkos::CudaHostPinnedSpace(), buf_comms_send[i]);
      h_buf_comms_recv[i] = Kokkos::create_mirror_view(Kokkos::CudaHostPinnedSpace(), buf_comms_recv[i]);
    }
  }

  int iswap;

  if (preprocess) {
    for(iswap = nswap - 1; iswap >= 0; iswap--) {

      int_1d_view_type list = Kokkos::subview(sendlist,iswap,Kokkos::ALL());

      // Pack buffer
      atom.pack_reverse(recvnum[iswap], firstrecv[iswap], buf_send);
      Kokkos::fence();

      // Exchange with another chare
      // If self, set recv buffer to send buffer
      if(sendchare[iswap] != index) {

        // Move data on device to host for communication
        h_buf_send = Kokkos::create_mirror_view(buf_send);
        h_buf_recv = Kokkos::create_mirror_view(buf_recv);
        Kokkos::deep_copy(h_buf_send, buf_send);

        // Send and suspend
        send1 = h_buf_send.data();
        send1_size = reverse_send_size[iswap] * sizeof(MMD_float);
        send1_chare = recvchare[iswap];
        recv1 = h_buf_recv.data();
        block_proxy[thisIndex].comms(iswap, CkCallbackResumeThread());

        // Move received data to device
        Kokkos::deep_copy(h2d_instance, buf_recv, h_buf_recv);

        buf = buf_recv;
      } else buf = buf_send;

      // unpack buffer
      atom.unpack_reverse(sendnum[iswap], list, buf);
      Kokkos::fence();
    }
  } else {
#if !defined PACK_UNPACK_COMPUTE
    // Enforce compute -> pack dependency
    cudaEvent_t dep_event_1;
    hapiCheck(cudaEventCreateWithFlags(&dep_event_1, cudaEventDisableTiming));
    hapiCheck(cudaEventRecord(dep_event_1, compute_instance.cuda_stream()));
    hapiCheck(cudaStreamWaitEvent(pack_instance.cuda_stream(), dep_event_1, 0));
#endif

    // XXX: Don't need to iterate backwards?
    for (iswap = nswap-1; iswap >= 0; iswap--) {
      int_1d_view_type list = Kokkos::subview(sendlist,iswap,Kokkos::ALL());

      // Pack and move buffers to host
      atom.pack_reverse(recvnum[iswap], firstrecv[iswap], buf_comms_send[iswap]);

#ifdef PACK_UNPACK_COMPUTE
      // Enforce compute -> d2h dependency
      cudaEvent_t dep_event;
      hapiCheck(cudaEventCreateWithFlags(&dep_event, cudaEventDisableTiming));
      hapiCheck(cudaEventRecord(dep_event, compute_instance.cuda_stream()));
      hapiCheck(cudaStreamWaitEvent(d2h_instance.cuda_stream(), dep_event, 0));
#else
      // Enforce pack -> d2h dependency
      cudaEvent_t dep_event;
      hapiCheck(cudaEventCreateWithFlags(&dep_event, cudaEventDisableTiming));
      hapiCheck(cudaEventRecord(dep_event, pack_instance.cuda_stream()));
      hapiCheck(cudaStreamWaitEvent(d2h_instance.cuda_stream(), dep_event, 0));
#endif

      if (sendchare[iswap] != index) {
        Kokkos::deep_copy(d2h_instance, h_buf_comms_send[iswap], buf_comms_send[iswap]);
      }
    }

#ifdef CUDA_SYNC
    d2h_instance.fence();
#else
    suspend(d2h_instance);
#endif

    // Unpack self
    for (iswap = nswap-1; iswap >= 0; iswap--) {
      int_1d_view_type list = Kokkos::subview(sendlist,iswap,Kokkos::ALL());

      if (sendchare[iswap] == index) {
        atom.unpack_reverse(sendnum[iswap], list, buf_comms_send[iswap]);
      }
    }

    // All buffers copied to host, send to neighbors
    // After receiving, move buffers to device and unpack
    atom_p = &atom;
    block_proxy[thisIndex].comm_rev_all(CkCallbackResumeThread());

#if !defined PACK_UNPACK_COMPUTE
    // Enforce unpack -> compute dependency
    cudaEvent_t dep_event;
    hapiCheck(cudaEventCreateWithFlags(&dep_event, cudaEventDisableTiming));
    hapiCheck(cudaEventRecord(dep_event, unpack_instance.cuda_stream()));
    hapiCheck(cudaStreamWaitEvent(compute_instance.cuda_stream(), dep_event, 0));
#endif
  }

  Kokkos::Profiling::popRegion();
}

/* exchange:
   move atoms to correct proc boxes
   send out atoms that have left my box, receive ones entering my box
   this routine called before every reneighboring
   atoms exchanged with all 6 stencil neighbors
*/

void Comm::exchange(Atom &atom_, bool preprocess)
{
  //NVTXTracer("Comm::exchange", NVTXColor::WetAsphalt);
  Kokkos::Profiling::pushRegion("exchange");
  atom = atom_;

  /* enforce PBC */

  atom.pbc();

  // Create host mirrors for integrate loop
  if (!preprocess) {
    if (!h_exc_alloc) {
      h_exc_alloc = true;
      h_exc_sendflag = Kokkos::create_mirror_view(Kokkos::CudaHostPinnedSpace(), exc_sendflag);
      h_exc_copylist = Kokkos::create_mirror_view(Kokkos::CudaHostPinnedSpace(), exc_copylist);
      h_exc_sendlist = Kokkos::create_mirror_view(Kokkos::CudaHostPinnedSpace(), exc_sendlist);
    }
    if (!h_buf_alloc) {
      h_buf_alloc = true;
      h_buf_send = Kokkos::create_mirror_view(Kokkos::CudaHostPinnedSpace(), buf_send);
      h_buf_recv = Kokkos::create_mirror_view(Kokkos::CudaHostPinnedSpace(), buf_recv);
    }
  }

  /* loop over dimensions */

  for(idim = 0; idim < 3; idim++) {
    /* only exchange if more than one proc in this dimension */

    if(charegrid[idim] == 1) continue;

    /* fill buffer with atoms leaving my box
       when atom is deleted, fill it in with last atom */

    nsend = 0;

    if(idim == 0) {
      lo = atom.box.xlo;
      hi = atom.box.xhi;
    } else if(idim == 1) {
      lo = atom.box.ylo;
      hi = atom.box.yhi;
    } else {
      lo = atom.box.zlo;
      hi = atom.box.zhi;
    }

    x = atom.x;

    nlocal = atom.nlocal;

    if (exc_sendflag.extent(0)<nlocal) {
      CmiEnforce(preprocess);
      Kokkos::resize(exc_sendflag,nlocal);
    }

    count.h_view(0) = exc_sendlist.extent(0);

    while (count.h_view(0) >= exc_sendlist.extent(0)) {
      count.h_view(0) = 0;
      count.modify<HostType>();
      count.sync<DeviceType>();

      Kokkos::parallel_for(Kokkos::RangePolicy<TagExchangeSendlist>(0,nlocal), *this);
      Kokkos::fence();

      count.modify<DeviceType>();
      count.sync<HostType>();
      if ((count.h_view(0)>=exc_sendlist.extent(0)) ||
          (count.h_view(0)>=exc_copylist.extent(0)) ) {
        CmiEnforce(preprocess);
        Kokkos::resize(exc_sendlist,(count.h_view(0)+1)*1.1);
        Kokkos::resize(exc_copylist,(count.h_view(0)+1)*1.1);
        count.h_view(0)=exc_sendlist.extent(0);
      }
      if (count.h_view(0)*7>=maxsend) {
        CmiEnforce(preprocess);
        growsend(count.h_view(0));
      }
    }
    if (preprocess) {
      h_exc_sendflag = Kokkos::create_mirror_view(exc_sendflag);
      h_exc_copylist = Kokkos::create_mirror_view(exc_copylist);
      h_exc_sendlist = Kokkos::create_mirror_view(exc_sendlist);
    }

    Kokkos::deep_copy(h_exc_sendflag,exc_sendflag);
    Kokkos::deep_copy(h_exc_copylist,exc_copylist);
    Kokkos::deep_copy(h_exc_sendlist,exc_sendlist);

    int sendpos = nlocal-1;
    nlocal -= count.h_view(0);
    for(int i = 0; i < count.h_view(0); i++) {
      if (h_exc_sendlist(i)<nlocal) {
        while (h_exc_sendflag(sendpos)) sendpos--;
        h_exc_copylist(i) = sendpos;
        sendpos--;
      } else
        h_exc_copylist(i) = -1;
    }
    Kokkos::deep_copy(exc_copylist,h_exc_copylist);

    Kokkos::parallel_for(Kokkos::RangePolicy<TagExchangePack>(0,count.h_view(0)), *this);

    atom.nlocal -= count.h_view(0);
    Kokkos::fence();

    nsend = count.h_view(0) * 7;

    send1 = static_cast<void*>(&nsend);
    send1_size = sizeof(int);
    send1_chare = chareneigh[idim][0];
    send2 = static_cast<void*>(&nsend);
    send2_size = sizeof(int);
    send2_chare = chareneigh[idim][1];
    recv1 = static_cast<void*>(&nrecv1);
    recv2 = static_cast<void*>(&nrecv2);
    block_proxy[thisIndex].exchange_1(idim, CkCallbackResumeThread());

    /*
    MPI_Sendrecv(&nsend, 1, MPI_INT, chareneigh[idim][0], 0,
                 &nrecv1, 1, MPI_INT, chareneigh[idim][1], 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    nrecv = nrecv1;

    if(charegrid[idim] > 2) {
      MPI_Sendrecv(&nsend, 1, MPI_INT, chareneigh[idim][1], 0,
                   &nrecv2, 1, MPI_INT, chareneigh[idim][0], 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      nrecv += nrecv2;
    }
    */

    if (nrecv > maxrecv) {
      CmiEnforce(preprocess);
      growrecv(nrecv);
    }
    Kokkos::fence();

    // Move data on device to host for communication
    if (preprocess) {
      h_buf_send = Kokkos::create_mirror_view(buf_send);
      h_buf_recv = Kokkos::create_mirror_view(buf_recv);
    } else {
      CmiEnforce(h_buf_alloc);
    }
    Kokkos::deep_copy(h_buf_send, buf_send);

    send1 = static_cast<void*>(h_buf_send.data());
    send1_size = nsend * sizeof(MMD_float);
    send1_chare = chareneigh[idim][0];
    send2 = static_cast<void*>(h_buf_send.data());
    send2_size = nsend * sizeof(MMD_float);
    send2_chare = chareneigh[idim][1];
    recv1 = h_buf_recv.data();
    recv2 = h_buf_recv.data() + nrecv1;
    block_proxy[thisIndex].exchange_2(idim, CkCallbackResumeThread());

    // Move received data to device
    Kokkos::deep_copy(buf_recv, h_buf_recv);

    /*
    MPI_Datatype type = (sizeof(MMD_float) == 4) ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Sendrecv(buf_send.data(), nsend, type, chareneigh[idim][0], 0,
                 buf_recv.data(), nrecv1, type, chareneigh[idim][1], 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if(charegrid[idim] > 2) {
      MPI_Sendrecv(buf_send.data(), nsend, type, chareneigh[idim][1], 0,
                   buf_recv.data()+nrecv1, nrecv2, type, chareneigh[idim][0], 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    */

    nrecv_atoms = nrecv / 7;

    /* check incoming atoms to see if they are in my box
       if they are, add to my list */

    nrecv = 0;

    Kokkos::parallel_reduce(Kokkos::RangePolicy<TagExchangeCountRecv>(0,nrecv_atoms), *this, nrecv);

    nlocal = atom.nlocal;

    if(nrecv_atoms>0)
    atom.nlocal += nrecv;

    count.h_view(0) = nlocal;
    count.modify<HostType>();
    count.sync<DeviceType>();

    if(atom.nlocal>=atom.nmax)
      atom.growarray();

    Kokkos::parallel_for(Kokkos::RangePolicy<TagExchangeUnpack>(0,nrecv_atoms), *this);
    Kokkos::fence();

  }
  atom_ = atom;
  Kokkos::Profiling::popRegion();
}

KOKKOS_INLINE_FUNCTION
void Comm::operator() (TagExchangeSendlist, const int& i) const {
  if (x(i,idim) < lo || x(i,idim) >= hi) {
    const int mysend=Kokkos::atomic_fetch_add(&count.d_view(0),1);
    if(mysend<exc_sendlist.extent(0)) {
      exc_sendlist(mysend) = i;
      exc_sendflag(i) = 1;
    }
  } else
    exc_sendflag(i) = 0;
}
KOKKOS_INLINE_FUNCTION
void Comm::operator() (TagExchangePack, const int& i ) const {
  atom.pack_exchange(exc_sendlist(i),&buf_send[7*i]);

  if(exc_copylist(i) > 0)
    atom.copy(exc_copylist(i),exc_sendlist(i));
}
KOKKOS_INLINE_FUNCTION
void Comm::operator() (TagExchangeCountRecv, const int& i, int& sum) const {
  const MMD_float value = buf_recv[i * 7 + idim];
  if(value >= lo && value < hi)
    sum++;
}
KOKKOS_INLINE_FUNCTION
void Comm::operator() (TagExchangeUnpack, const int& i ) const {
  double value = buf_recv[i * 7 + idim];

  if(value >= lo && value < hi)
    atom.unpack_exchange(Kokkos::atomic_fetch_add(&count.d_view(0),1), &buf_recv[i * 7]);
}

/* borders:
   make lists of nearby atoms to send to neighboring procs at every timestep
   one list is created for every swap that will be made
   as list is made, actually do swaps
   this does equivalent of a communicate (so don't need to explicitly
     call communicate routine on reneighboring timestep)
   this routine is called before every reneighboring
*/

void Comm::borders(Atom &atom_, bool preprocess)
{
  //NVTXTracer("Comm::borders", NVTXColor::Carrot);
  Kokkos::Profiling::pushRegion("Comm::borders");

  // Create host mirrors for integrate loop
  if (!preprocess && !h_buf_alloc) {
    h_buf_alloc = true;
    h_buf_send = Kokkos::create_mirror_view(Kokkos::CudaHostPinnedSpace(), buf_send);
    h_buf_recv = Kokkos::create_mirror_view(Kokkos::CudaHostPinnedSpace(), buf_recv);
  }

  atom = atom_;
  int ineed, nsend, nrecv, nfirst, nlast;

  /* erase all ghost atoms */

  atom.nghost = 0;

  /* do swaps over all 3 dimensions */

  iswap = 0;


  if(atom.nlocal > maxnlocal) {
    send_flag = int_1d_view_type("Comm::sendflag",atom.nlocal);
    maxnlocal = atom.nlocal;
  }

  for(idim = 0; idim < 3; idim++) {
    nlast = 0;

    for(ineed = 0; ineed < 2 * need[idim]; ineed++) {

      // find atoms within slab boundaries lo/hi using <= and >=
      // check atoms between nfirst and nlast
      //   for first swaps in a dim, check owned and ghost
      //   for later swaps in a dim, only check newly arrived ghosts
      // store sent atom indices in list for use in future timesteps

      lo = slablo[iswap];
      hi = slabhi[iswap];
      pbc_flags[0] = pbc_any[iswap];
      pbc_flags[1] = pbc_flagx[iswap];
      pbc_flags[2] = pbc_flagy[iswap];
      pbc_flags[3] = pbc_flagz[iswap];

      x = atom.x;

      if(ineed % 2 == 0) {
        nfirst = nlast;
        nlast = atom.nlocal + atom.nghost;
      }

      nsend = 0;

      count.h_view(0) = 0;
      count.modify<HostType>();
      count.sync<DeviceType>();

      send_count = count.d_view;

      Kokkos::parallel_for(Kokkos::RangePolicy<TagBorderSendlist>(nfirst,nlast),*this);

      count.modify<DeviceType>();
      count.sync<HostType>();

      nsend = count.h_view(0);
      if(nsend > exc_sendlist.extent(0)) {
        CmiEnforce(preprocess);
        Kokkos::resize(exc_sendlist , nsend);

        growlist(iswap, nsend);

        count.h_view(0) = 0;
        count.modify<HostType>();
        count.sync<DeviceType>();

        Kokkos::parallel_for(Kokkos::RangePolicy<TagBorderSendlist>(nfirst,nlast),*this);

        count.modify<DeviceType>();
        count.sync<HostType>();
      }

      if(nsend * 4 > maxsend) {
        CmiEnforce(preprocess);
        growsend(nsend * 4);
      }

      Kokkos::parallel_for(Kokkos::RangePolicy<TagBorderPack>(0,nsend),*this);
      Kokkos::fence();
      // swap atoms with other proc
      // put incoming ghosts at end of my atom arrays
      // if swapping with self, simply copy, no messages


        if(sendchare[iswap] != index) {
          send1 = static_cast<void*>(&nsend);
          send1_size = sizeof(int);
          send1_chare = sendchare[iswap];
          recv1 = static_cast<void*>(&nrecv);
          block_proxy[thisIndex].borders_1(iswap, CkCallbackResumeThread());

          if(nrecv * atom.border_size > maxrecv) {
            CmiEnforce(preprocess);
            growrecv(nrecv * atom.border_size);
          }

          // Move data on device to host for communication
          if (preprocess) {
            h_buf_send = Kokkos::create_mirror_view(buf_send);
            h_buf_recv = Kokkos::create_mirror_view(buf_recv);
          } else {
            CmiEnforce(h_buf_alloc);
          }
          Kokkos::deep_copy(h_buf_send, buf_send);

          send1 = static_cast<void*>(h_buf_send.data());
          send1_size = nsend * atom.border_size * sizeof(MMD_float);
          send1_chare = sendchare[iswap];
          recv1 = h_buf_recv.data();
          block_proxy[thisIndex].borders_2(iswap, CkCallbackResumeThread());

          // Move received data to device
          Kokkos::deep_copy(buf_recv, h_buf_recv);

          buf = buf_recv;
        } else {
          nrecv = nsend;
          buf = buf_send;
        }

      // unpack buffer

      n = atom.nlocal + atom.nghost;

      while(n + nrecv > atom.nmax) atom.growarray();

      x = atom.x;

      Kokkos::parallel_for(Kokkos::RangePolicy<TagBorderUnpack>(0,nrecv),*this);
      Kokkos::fence();

      // set all pointers & counters

        sendnum[iswap] = nsend;
        recvnum[iswap] = nrecv;
        comm_send_size[iswap] = nsend * atom.comm_size;
        comm_recv_size[iswap] = nrecv * atom.comm_size;
        reverse_send_size[iswap] = nrecv * atom.reverse_size;
        reverse_recv_size[iswap] = nsend * atom.reverse_size;
        firstrecv[iswap] = atom.nlocal + atom.nghost;
        atom.nghost += nrecv;

      iswap++;
    }
  }

  /* insure buffers are large enough for reverse comm */

  int max1, max2;
  max1 = max2 = 0;

  for(iswap = 0; iswap < nswap; iswap++) {
    max1 = MAX(max1, reverse_send_size[iswap]);
    max2 = MAX(max2, reverse_recv_size[iswap]);
  }

  if(max1 > maxsend) {
    CmiEnforce(preprocess);
    growsend(max1);
  }

  if(max2 > maxrecv) {
    CmiEnforce(preprocess);
    growrecv(max2);
  }
  atom_ = atom;

  Kokkos::Profiling::popRegion();
}

KOKKOS_INLINE_FUNCTION
void Comm::operator() (TagBorderSendlist, const int& i) const {
  if(x(i,idim) >= lo && x(i,idim) <= hi) {
    const int nsend = (send_count(0)+=1)-1;
    if(nsend < exc_sendlist.extent(0)) {
      exc_sendlist[nsend] = i;
    }
  }
}

KOKKOS_INLINE_FUNCTION
void Comm::operator() (TagBorderPack, const int& k) const {
  atom.pack_border(exc_sendlist(k), &buf_send[k * 4], pbc_flags);
  sendlist(iswap,k) = exc_sendlist(k);
}

KOKKOS_INLINE_FUNCTION
void Comm::operator() (TagBorderUnpack, const int& i) const {
  atom.unpack_border(n + i, &buf[i * 4]);
}

/* realloc the size of the send buffer as needed with BUFFACTOR & BUFEXTRA */

void Comm::growsend(int n)
{
  Kokkos::resize(buf_send,static_cast<int>(BUFFACTOR * n) + BUFEXTRA);
  maxsend = static_cast<int>(BUFFACTOR * n);
}

/* free/malloc the size of the recv buffer as needed with BUFFACTOR */

void Comm::growrecv(int n)
{
  maxrecv = static_cast<int>(BUFFACTOR * n) + BUFEXTRA;
  buf_recv = float_1d_view_type("Comm::buf_recv",maxrecv);;
}

/* realloc the size of the iswap sendlist as needed with BUFFACTOR */

void Comm::growlist(int iswap, int n)
{
  if(n<=maxsendlist[iswap]) return;
  int maxswap = sendlist.extent(0);
  Kokkos::resize(sendlist,sendlist.extent(0),BUFFACTOR * n + BUFEXTRA);
  for(int iswaps = 0; iswaps < maxswap; iswaps++) {
    maxsendlist[iswaps] = static_cast<int>(BUFFACTOR * n);
  }
}

void Comm::suspend(Kokkos::Cuda instance) {
  resume_cb = new CkCallbackResumeThread();
  hapiAddCallback(instance.cuda_stream(), resume_cb);
  delete resume_cb;
}
