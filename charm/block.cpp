#include "miniMD.decl.h"
#include "block.decl.h"
#include "block.h"
#include "force_eam.h"
#include "force_lj.h"

/* readonly */ extern CProxy_Main main_proxy;
/* readonly */ extern CProxy_Block block_proxy;
/* readonly */ extern CProxy_Comm comm_proxy;
/* readonly */ extern int num_chares;

/* readonly */ extern std::string input_file;
/* readonly */ extern int num_threads;
/* readonly */ extern int teams;
/* readonly */ extern int num_steps;
/* readonly */ extern int system_size;
/* readonly */ extern int nx;
/* readonly */ extern int ny;
/* readonly */ extern int nz;
/* readonly */ extern int ntypes;
/* readonly */ extern int neighbor_size;
/* readonly */ extern int halfneigh;
/* readonly */ extern int team_neigh;
/* readonly */ extern int use_sse;
/* readonly */ extern int check_safeexchange;
/* readonly */ extern int do_safeexchange;
/* readonly */ extern int sort;
/* readonly */ extern int yaml_output;
/* readonly */ extern int yaml_screen;
/* readonly */ extern int ghost_newton;
/* readonly */ extern int in_nx;
/* readonly */ extern int in_ny;
/* readonly */ extern int in_nz;
/* readonly */ extern MMD_float in_t_request;
/* readonly */ extern MMD_float in_rho;
/* readonly */ extern int in_units;
/* readonly */ extern ForceStyle in_forcetype;
/* readonly */ extern MMD_float in_epsilon;
/* readonly */ extern MMD_float in_sigma;
/* readonly */ extern std::string in_datafile;
/* readonly */ extern int in_ntimes;
/* readonly */ extern MMD_float in_dt;
/* readonly */ extern int in_neigh_every;
/* readonly */ extern MMD_float in_force_cut;
/* readonly */ extern MMD_float in_neigh_cut;
/* readonly */ extern int in_thermo_nstat;

extern void create_box(Atom& atom, int nx, int ny, int nz, double rho);
extern int create_atoms(Atom& atom, int nx, int ny, int nz, double rho,
    Kokkos::Cuda comm_instance);
extern void create_velocity_1(Atom &atom, double& vxtot, double& vytot,
    double& vztot, Kokkos::Cuda comm_instance);
extern void create_velocity_2(double t_request, Atom &atom, Thermo &thermo,
    double vxtot, double vytot, double vztot, Kokkos::Cuda comm_instance);

Block::Block() : atom(ntypes), neighbor(ntypes), integrate(), thermo(),
  comm(nullptr), force(nullptr) {}

void Block::init() {
  // Save pointer to Comm bound array element
  comm = comm_proxy(thisIndex).ckLocal();

  // Create force object
  if (in_forcetype == FORCEEAM) {
    force = (Force*) new ForceEAM(ntypes);
  } else if (in_forcetype == FORCELJ) {
    force = (Force*) new ForceLJ(ntypes);
  }

  // Create CUDA streams (higher priority for communication stream)
  cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault, 0);
  cudaStreamCreateWithPriority(&comm_stream, cudaStreamDefault, -1);

  // Create separate execution instances with CUDA streams
  compute_instance = Kokkos::Cuda(compute_stream);
  comm_instance = Kokkos::Cuda(comm_stream);
  atom.compute_instance = compute_instance;
  atom.comm_instance = comm_instance;
  neighbor.compute_instance = compute_instance;
  neighbor.comm_instance = comm_instance;
  integrate.compute_instance = compute_instance;
  integrate.comm_instance = comm_instance;
  thermo.compute_instance = compute_instance;
  thermo.comm_instance = comm_instance;
  comm->compute_instance = compute_instance;
  comm->comm_instance = comm_instance;
  force->compute_instance = compute_instance;
  force->comm_instance = comm_instance;

  // Create CUDA events used to preserve dependencies between streams
  cudaEventCreateWithFlags(&compute_event, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&comm_event, cudaEventDisableTiming);
  atom.compute_event = compute_event;
  atom.comm_event = comm_event;
  neighbor.compute_event = compute_event;
  neighbor.comm_event = comm_event;
  integrate.compute_event = compute_event;
  integrate.comm_event = comm_event;
  thermo.compute_event = compute_event;
  thermo.comm_event = comm_event;
  comm->compute_event = compute_event;
  comm->comm_event = comm_event;
  force->compute_event = compute_event;
  force->comm_event = comm_event;

  // Store chare index
  integrate.index = thisIndex;
  force->index = thisIndex;

  if (in_forcetype == FORCELJ) {
    float_1d_view_type d_epsilon("ForceLJ::epsilon", ntypes*ntypes);
    float_1d_host_view_type h_epsilon = Kokkos::create_mirror_view(d_epsilon);
    force->epsilon = d_epsilon;
    force->epsilon_scalar = in_epsilon;

    float_1d_view_type d_sigma6("ForceLJ::sigma6", ntypes*ntypes);
    float_1d_host_view_type h_sigma6 = Kokkos::create_mirror_view(d_sigma6);
    force->sigma6 = d_sigma6;

    float_1d_view_type d_sigma("ForceLJ::sigma", ntypes*ntypes);
    float_1d_host_view_type h_sigma = Kokkos::create_mirror_view(d_sigma);
    force->sigma = d_sigma;
    force->sigma_scalar = in_sigma;

    for (int i=0; i< ntypes * ntypes; i++) {
      h_epsilon[i] = in_epsilon;
      h_sigma[i] = in_sigma;
      h_sigma6[i] = in_sigma*in_sigma*in_sigma*in_sigma*in_sigma*in_sigma;
      if (i < MAX_STACK_TYPES * MAX_STACK_TYPES) {
        force->epsilon_s[i] = h_epsilon[i];
        force->sigma6_s[i] = h_sigma6[i];
      }
    }

    Kokkos::deep_copy(d_epsilon, h_epsilon);
    Kokkos::deep_copy(d_sigma6, h_sigma6);
    Kokkos::deep_copy(d_sigma, h_sigma);
  }

  neighbor.ghost_newton = ghost_newton;
  comm->check_safeexchange = check_safeexchange;
  comm->do_safeexchange = do_safeexchange;
  force->use_sse = use_sse;
  neighbor.halfneigh = halfneigh;
  neighbor.team_neigh_build = team_neigh;
  if (halfneigh < 0) force->use_oldcompute = 1;

#ifdef VARIANT_REFERENCE
  if (use_sse) {
    if (thisIndex == 0) {
      CkPrintf("ERROR: Trying to run with -sse with miniMD reference version. "
          "Use SSE variant instead. Exiting.\n");
      CkExit();
    }
  }
#endif

  if (neighbor_size > 0) {
    neighbor.nbinx = neighbor_size;
    neighbor.nbiny = neighbor_size;
    neighbor.nbinz = neighbor_size;
  }

  if (neighbor_size < 0 && in_datafile.empty()) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_ROCM)
    MMD_float neighscale = 0.6;
#else
    MMD_float neighscale = 5.0 / 6.0;
#endif
     neighbor.nbinx = neighscale * in_nx;
     neighbor.nbiny = neighscale * in_ny;
     neighbor.nbinz = neighscale * in_nz;
   }

   if (neighbor_size < 0 && !in_datafile.empty())
     neighbor.nbinx = -1;

   if (neighbor.nbinx == 0) neighbor.nbinx = 1;
   if (neighbor.nbiny == 0) neighbor.nbiny = 1;
   if (neighbor.nbinz == 0) neighbor.nbinz = 1;

   integrate.ntimes = in_ntimes;
   integrate.dt = in_dt;
   integrate.sort_every = sort>0?sort:(sort<0?in_neigh_every:0);
   neighbor.every = in_neigh_every;
   neighbor.cutneigh = in_neigh_cut;
   force->cutforce = in_force_cut;
   thermo.nstat = in_thermo_nstat;

   if (thisIndex == 0)
     CkPrintf("# Create System:\n");

   if (!in_datafile.empty()) {
     if (thisIndex == 0) {
       CkPrintf("Lammps data file not yet supported\n");
       CkExit();
     }
   } else {
     create_box(atom, in_nx, in_ny, in_nz, in_rho);

     comm->setup(neighbor.cutneigh, atom);

     neighbor.setup(atom);

     integrate.setup();

     force->setup();

     if (in_forcetype == FORCEEAM) atom.mass = force->mass;

     create_atoms(atom, in_nx, in_ny, in_nz, in_rho, comm_instance);

     thermo.setup(in_rho, integrate, atom, in_units);

     create_velocity_1(atom, vtot[0], vtot[1], vtot[2], comm_instance);
   }

   CkCallback cb(CkCallback(CkReductionTarget(Main, reduceVelocity), main_proxy));
   contribute(3*sizeof(double), vtot, CkReduction::set, cb);
}

void Block::contCreateVelocity(double vxtot, double vytot, double vztot) {
  if (in_datafile.empty()) {
    create_velocity_2(in_t_request, atom, thermo, vxtot, vytot, vztot,
        comm_instance);
  }

  printConfig();

  contribute(CkCallback(CkReductionTarget(Main, blockInitialized), main_proxy));
}

void Block::printConfig() {
  if (thisIndex == 0) {
    CkPrintf("# Done .... \n");
    CkPrintf("# Charm++ + Kokkos MiniMD output ...\n");
    CkPrintf("# Run Settings: \n");
    CkPrintf("\t# Chares: %i\n", num_chares);
    CkPrintf("\t# Host Threads: %i\n", Kokkos::HostSpace::execution_space::concurrency());
    CkPrintf("\t# Inputfile: %s\n", input_file.c_str());
    CkPrintf("\t# Datafile: %s\n", in_datafile.empty() ? "None" : in_datafile.c_str());
    CkPrintf("# Physics Settings: \n");
    CkPrintf("\t# ForceStyle: %s\n", in_forcetype == FORCELJ ? "LJ" : "EAM");
    CkPrintf("\t# Force Parameters: %2.2lf %2.2lf\n",in_epsilon,in_sigma);
    CkPrintf("\t# Units: %s\n", in_units == 0 ? "LJ" : "METAL");
    CkPrintf("\t# Atoms: %i\n", atom.natoms);
    CkPrintf("\t# Atom types: %i\n", atom.ntypes);
    CkPrintf("\t# System size: %2.2lf %2.2lf %2.2lf (unit cells: %i %i %i)\n", atom.box.xprd, atom.box.yprd, atom.box.zprd, in_nx, in_ny, in_nz);
    CkPrintf("\t# Density: %lf\n", in_rho);
    CkPrintf("\t# Force cutoff: %lf\n", force->cutforce);
    CkPrintf("\t# Timestep size: %lf\n", integrate.dt);
    CkPrintf("# Technical Settings: \n");
    CkPrintf("\t# Neigh cutoff: %lf\n", neighbor.cutneigh);
    CkPrintf("\t# Half neighborlists: %i\n", neighbor.halfneigh);
    CkPrintf("\t# Team neighborlist construction: %i\n", neighbor.team_neigh_build);
    CkPrintf("\t# Neighbor bins: %i %i %i\n", neighbor.nbinx, neighbor.nbiny, neighbor.nbinz);
    CkPrintf("\t# Neighbor frequency: %i\n", neighbor.every);
    CkPrintf("\t# Sorting frequency: %i\n", integrate.sort_every);
    CkPrintf("\t# Thermo frequency: %i\n", thermo.nstat);
    CkPrintf("\t# Ghost Newton: %i\n", ghost_newton);
    CkPrintf("\t# Use intrinsics: %i\n", force->use_sse);
    CkPrintf("\t# Do safe exchange: %i\n", comm->do_safeexchange);
    CkPrintf("\t# Size of float: %i\n\n", (int) sizeof(MMD_float));
  }
}

#include "block.def.h"
