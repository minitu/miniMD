#include "ljs_kokkos.h"
#include "ljs_kokkos_api.h"
#include "atom.h"
#include "neighbor.h"
#include "integrate.h"
#include "thermo.h"
#include "comm.h"
#include "force.h"
#include "force_lj.h"
#include "force_eam.h"

/* readonly */ extern int num_chares;
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
extern int create_atoms(Atom& atom, int nx, int ny, int nz, double rho);
extern void create_velocity(double t_request, Atom& atom, Thermo& thermo);

void kokkosInitialize(int num_threads, int teams, int device) {
  Kokkos::InitArguments args_kokkos;
  args_kokkos.num_threads = num_threads;
  args_kokkos.num_numa = teams;
  args_kokkos.device_id = device;
  Kokkos::initialize(args_kokkos);
}

void kokkosFinalize() {
  Kokkos::finalize();
}

// Can't expose this to Charm++ code because of Kokkos code that needs to be
// compiled by nvcc_wrapper and not charmc
struct BlockKokkos {
  int index;

  Atom atom;
  Neighbor neighbor;
  Integrate integrate;
  Thermo thermo;
  Comm comm;
  Force* force;

  cudaStream_t compute_stream;
  cudaStream_t comm_stream;

  cudaEvent_t compute_event;
  cudaEvent_t comm_event;

  Kokkos::Cuda compute_instance;
  Kokkos::Cuda comm_instance;

  BlockKokkos(int index_, cudaStream_t compute_stream_, cudaStream_t comm_stream_,
      cudaEvent_t compute_event_, cudaEvent_t comm_event_) :
    index(index_), compute_stream(compute_stream_), comm_stream(comm_stream_),
    compute_event(compute_event_), comm_event(comm_event_),
    atom(ntypes), neighbor(ntypes), integrate(), thermo(), comm(index_) {

    force = NULL;
    if (in_forcetype == FORCEEAM) {
      force = (Force*) new ForceEAM(ntypes);
    } else if (in_forcetype == FORCELJ) {
      force = (Force*) new ForceLJ(ntypes);
    }

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
    force->compute_instance = compute_instance;
    force->comm_instance = comm_instance;

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

      Kokkos::deep_copy(comm_instance, d_epsilon, h_epsilon);
      Kokkos::deep_copy(comm_instance, d_sigma6, h_sigma6);
      Kokkos::deep_copy(comm_instance, d_sigma, h_sigma);
    }

    neighbor.ghost_newton = ghost_newton;
    comm.check_safeexchange = check_safeexchange;
    comm.do_safeexchange = do_safeexchange;
    force->use_sse = use_sse;
    neighbor.halfneigh = halfneigh;
    neighbor.team_neigh_build = team_neigh;
    if (halfneigh < 0) force->use_oldcompute = 1;

#ifdef VARIANT_REFERENCE
    if (use_sse) {
      if (index == 0) {
        printf("ERROR: Trying to run with -sse with miniMD reference version. "
            "Use SSE variant instead. Exiting.\n");
      }
      exit(0);
    }
#endif

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

    if (index == 0)
      printf("# Create System:\n");

    if (!in_datafile.empty()) {
      if (index == 0) {
        printf("Lammps data file not yet supported\n");
      }
      exit(0);

      /* TODO
      read_lammps_data(atom, comm, neighbor, integrate, thermo,
          in_datafile.c_str(), in_units);
      MMD_float volume = atom.box.xprd * atom.box.yprd * atom.box.zprd;
      in_rho = 1.0 * atom.natoms / volume;
      force->setup();

      if (in_forcetype == FORCEEAM) atom.mass = force->mass;
      */
    } else {
      create_box(atom, in_nx, in_ny, in_nz, in_rho);

      comm.setup(neighbor.cutneigh, atom);

      neighbor.setup(atom);

      /*
      integrate.setup();

      force->setup();

      if (in_forcetype == FORCEEAM) atom.mass = force->mass;

      create_atoms(atom, in_nx, in_ny, in_nz, in_rho);
      thermo.setup(in_rho, integrate, atom, in_units);

      create_velocity(in_t_request, atom, thermo);
      */
    }

    if (index == 0)
      printf("# Done .... \n");

    if (index == 0) {
      fprintf(stdout, "# Charm++ + Kokkos MiniMD output ...\n");
      fprintf(stdout, "# Run Settings: \n");
      fprintf(stdout, "\t# Chares: %i\n", num_chares);
      fprintf(stdout, "\t# Host Threads: %i\n", Kokkos::HostSpace::execution_space::concurrency());
      fprintf(stdout, "\t# Datafile: %s\n", in_datafile.empty() ? "None" : in_datafile.c_str());
      fprintf(stdout, "# Physics Settings: \n");
      fprintf(stdout, "\t# ForceStyle: %s\n", in_forcetype == FORCELJ ? "LJ" : "EAM");
      fprintf(stdout, "\t# Force Parameters: %2.2lf %2.2lf\n",in_epsilon,in_sigma);
      fprintf(stdout, "\t# Units: %s\n", in_units == 0 ? "LJ" : "METAL");
      fprintf(stdout, "\t# Atoms: %i\n", atom.natoms);
      fprintf(stdout, "\t# Atom types: %i\n", atom.ntypes);
      fprintf(stdout, "\t# System size: %2.2lf %2.2lf %2.2lf (unit cells: %i %i %i)\n", atom.box.xprd, atom.box.yprd, atom.box.zprd, in_nx, in_ny, in_nz);
      fprintf(stdout, "\t# Density: %lf\n", in_rho);
      fprintf(stdout, "\t# Force cutoff: %lf\n", force->cutforce);
      fprintf(stdout, "\t# Timestep size: %lf\n", integrate.dt);
      fprintf(stdout, "# Technical Settings: \n");
      fprintf(stdout, "\t# Neigh cutoff: %lf\n", neighbor.cutneigh);
      fprintf(stdout, "\t# Half neighborlists: %i\n", neighbor.halfneigh);
      fprintf(stdout, "\t# Team neighborlist construction: %i\n", neighbor.team_neigh_build);
      fprintf(stdout, "\t# Neighbor bins: %i %i %i\n", neighbor.nbinx, neighbor.nbiny, neighbor.nbinz);
      fprintf(stdout, "\t# Neighbor frequency: %i\n", neighbor.every);
      fprintf(stdout, "\t# Sorting frequency: %i\n", integrate.sort_every);
      fprintf(stdout, "\t# Thermo frequency: %i\n", thermo.nstat);
      fprintf(stdout, "\t# Ghost Newton: %i\n", ghost_newton);
      fprintf(stdout, "\t# Use intrinsics: %i\n", force->use_sse);
      fprintf(stdout, "\t# Do safe exchange: %i\n", comm.do_safeexchange);
      fprintf(stdout, "\t# Size of float: %i\n\n", (int) sizeof(MMD_float));
    }
  }
};

void blockNew(void** block, int index, cudaStream_t compute_stream,
    cudaStream_t comm_stream, cudaEvent_t compute_event, cudaEvent_t comm_event) {
  *block = (void*)new BlockKokkos(index, compute_stream, comm_stream, compute_event,
      comm_event);
}

void blockDelete(void* block) {
  delete (BlockKokkos*)block;
}
