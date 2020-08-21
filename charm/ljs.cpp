#include "miniMD.decl.h"
#include "pup_stl.h"
#include "hapi.h"

#include "ljs_kokkos.h"
#include "atom.h"
#include "neighbor.h"
#include "integrate.h"
#include "thermo.h"
#include "comm.h"
#include "force.h"
#include "force_lj.h"
#include "force_eam.h"

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_KokkosManager kokkos_proxy;
/* readonly */ CProxy_Block block_proxy;
/* readonly */ int num_chares;

/* readonly */ std::string input_file;
/* readonly */ int num_threads;
/* readonly */ int teams;
/* readonly */ int num_steps;
/* readonly */ int system_size;
/* readonly */ int nx;
/* readonly */ int ny;
/* readonly */ int nz;
/* readonly */ int ntypes;
/* readonly */ int neighbor_size;
/* readonly */ int halfneigh;
/* readonly */ int team_neigh;
/* readonly */ int use_sse;
/* readonly */ int check_safeexchange;
/* readonly */ int do_safeexchange;
/* readonly */ int sort;
/* readonly */ int yaml_output;
/* readonly */ int yaml_screen;
/* readonly */ int ghost_newton;
/* readonly */ int in_nx;
/* readonly */ int in_ny;
/* readonly */ int in_nz;
/* readonly */ MMD_float in_t_request;
/* readonly */ MMD_float in_rho;
/* readonly */ int in_units;
/* readonly */ ForceStyle in_forcetype;
/* readonly */ MMD_float in_epsilon;
/* readonly */ MMD_float in_sigma;
/* readonly */ std::string in_datafile;
/* readonly */ int in_ntimes;
/* readonly */ MMD_float in_dt;
/* readonly */ int in_neigh_every;
/* readonly */ MMD_float in_force_cut;
/* readonly */ MMD_float in_neigh_cut;
/* readonly */ int in_thermo_nstat;

extern int input(const char* filename, int& in_nx, int& in_ny, int& in_nz,
    MMD_float& in_t_request, MMD_float& in_rho, int& in_units,
    ForceStyle& in_forcetype, MMD_float& in_epsilon, MMD_float& in_sigma,
    std::string& in_datafile, int& in_ntimes, MMD_float& in_dt,
    int& in_neigh_every, MMD_float& in_force_cut, MMD_float& in_neigh_cut,
    int& in_thermo_nstat);
extern void create_box(Atom& atom, int nx, int ny, int nz, double rho);
extern int create_atoms(Atom& atom, int nx, int ny, int nz, double rho,
    Kokkos::Cuda comm_instance);
extern void create_velocity_1(Atom &atom, double& vxtot, double& vytot,
    double& vztot, Kokkos::Cuda comm_instance);
extern void create_velocity_2(double t_request, Atom &atom, Thermo &thermo,
    double vxtot, double vytot, double vztot, Kokkos::Cuda comm_instance);

class Main : public CBase_Main {
  Main_SDAG_CODE

public:
  Main(CkArgMsg* m) {
    // Default parameters
    num_chares = 4;
    num_threads = 1;
    teams = 1;
    num_steps = -1;
    system_size = -1;
    nx = -1; int ny = -1; int nz = -1;
    ntypes = 8;
    neighbor_size = -1;
    halfneigh = 1;
    team_neigh = 0;
    use_sse = 0;
    check_safeexchange = 0;
    do_safeexchange = 0;
    sort = -1;
    yaml_output = 0;
    yaml_screen = 0;
    ghost_newton = 1;

    // Process input file
    for (int i = 0; i < m->argc; i++) {
      if ((strcmp(m->argv[i], "-i") == 0) || (strcmp(m->argv[i], "--input_file") == 0)) {
        input_file = std::string(m->argv[++i]);
        continue;
      }
    }

    int error = 0;
    if (input_file.empty()) {
      error = input("../inputs/in.lj.miniMD", in_nx, in_ny, in_nz, in_t_request,
          in_rho, in_units, in_forcetype, in_epsilon, in_sigma, in_datafile,
          in_ntimes, in_dt, in_neigh_every, in_force_cut, in_neigh_cut,
          in_thermo_nstat);
    } else {
      error = input(input_file.c_str(), in_nx, in_ny, in_nz, in_t_request,
          in_rho, in_units, in_forcetype, in_epsilon, in_sigma, in_datafile,
          in_ntimes, in_dt, in_neigh_every, in_force_cut, in_neigh_cut,
          in_thermo_nstat);
    }

    if (error) {
      CkPrintf("ERROR: Failed to read input file\n");
      CkExit();
    }

    srand(5413);

    // Process other command line parameters
    for (int i = 0; i < m->argc; i++) {
      if ((strcmp(m->argv[i], "-c") == 0) || (strcmp(m->argv[i], "--num_chares") == 0)) {
        num_chares = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "-t") == 0) || (strcmp(m->argv[i], "--num_threads") == 0)) {
        num_threads = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "--teams") == 0)) {
        teams = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "-n") == 0) || (strcmp(m->argv[i], "--nsteps") == 0))  {
        num_steps = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "-s") == 0) || (strcmp(m->argv[i], "--size") == 0)) {
        system_size = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "-nx") == 0)) {
        nx = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "-ny") == 0)) {
        ny = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "-nz") == 0)) {
        nz = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "--ntypes") == 0)) {
        ntypes = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "-b") == 0) || (strcmp(m->argv[i], "--neigh_bins") == 0))  {
        neighbor_size = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "--half_neigh") == 0))  {
        halfneigh = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "--team_neigh") == 0))  {
        team_neigh = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "-sse") == 0))  {
        use_sse = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "--check_exchange") == 0))  {
        check_safeexchange = 1;
        continue;
      }

      if ((strcmp(m->argv[i], "--safe_exchange") == 0)) {
        do_safeexchange = 1;
        continue;
      }

      if ((strcmp(m->argv[i], "--sort") == 0))  {
        sort = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "-o") == 0) || (strcmp(m->argv[i], "--yaml_output") == 0))  {
        yaml_output = atoi(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "--yaml_screen") == 0))  {
        yaml_screen = 1;
        continue;
      }

      if ((strcmp(m->argv[i], "-f") == 0) || (strcmp(m->argv[i], "--data_file") == 0)) {
        in_datafile = std::string(m->argv[++i]);
        continue;
      }

      if ((strcmp(m->argv[i], "-u") == 0) || (strcmp(m->argv[i], "--units") == 0)) {
        in_units = strcmp(m->argv[++i], "metal") == 0 ? 1 : 0;
        continue;
      }

      if ((strcmp(m->argv[i], "-p") == 0) || (strcmp(m->argv[i], "--force") == 0)) {
        in_forcetype = strcmp(m->argv[++i], "eam") == 0 ? FORCEEAM : FORCELJ;
        continue;
      }

      if ((strcmp(m->argv[i], "-gn") == 0) || (strcmp(m->argv[i], "--ghost_newton") == 0)) {
        ghost_newton = atoi(m->argv[++i]);
        continue;
      }
    }

    if (in_forcetype == FORCEEAM && ghost_newton == 1) {
      CkPrintf("# EAM currently requires '--ghost_newton 0'; Changing setting now.\n");
      ghost_newton = 0;
    }

    if (num_steps > 0) in_ntimes = num_steps;
    if (system_size > 0) in_nx = in_ny = in_nz = system_size;

    if (nx > 0) {
      in_nx = nx;
      if (ny > 0) in_ny = ny;
      else if (system_size < 0) in_ny = nx;

      if (nz > 0) in_nz = nz;
      else if (system_size < 0) in_nz = nx;
    }

    // Create KokkosManagers on each process
    kokkos_proxy = CProxy_KokkosManager::ckNew();

    thisProxy.run();
  }
};

class KokkosManager : public CBase_KokkosManager {
public:
  KokkosManager() {
    // Initialize Kokkos
    Kokkos::InitArguments args_kokkos;
    args_kokkos.num_threads = num_threads;
    args_kokkos.num_numa = teams;
    args_kokkos.device_id = 0;
    Kokkos::initialize(args_kokkos);

    contribute(CkCallback(CkReductionTarget(Main, kokkosInitialized), main_proxy));
  }

  void finalize() {
    // Finalize Kokkos
    Kokkos::finalize();

    contribute(CkCallback(CkReductionTarget(Main, kokkosFinalized), main_proxy));
  }
};

class Block : public CBase_Block {
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

  double vtot[3];

public:
  Block() : atom(ntypes), neighbor(ntypes), comm(thisIndex) {
    // Create CUDA streams (higher priority for communication stream)
    cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault, 0);
    cudaStreamCreateWithPriority(&comm_stream, cudaStreamDefault, -1);

    // Create CUDA events used to preserve dependencies between streams
    cudaEventCreateWithFlags(&compute_event, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&comm_event, cudaEventDisableTiming);

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

      Kokkos::deep_copy(d_epsilon, h_epsilon);
      Kokkos::deep_copy(d_sigma6, h_sigma6);
      Kokkos::deep_copy(d_sigma, h_sigma);
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
      if (thisIndex == 0) {
        CkPrintf("ERROR: Trying to run with -sse with miniMD reference version. "
            "Use SSE variant instead. Exiting.\n");
        CkExit();
      }
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

    if (thisIndex == 0)
      CkPrintf("# Create System:\n");

    if (!in_datafile.empty()) {
      if (thisIndex == 0) {
        CkPrintf("Lammps data file not yet supported\n");
        CkExit();
      }

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

      integrate.setup();

      force->setup();

      if (in_forcetype == FORCEEAM) atom.mass = force->mass;

      create_atoms(atom, in_nx, in_ny, in_nz, in_rho, comm_instance);

      thermo.setup(in_rho, integrate, atom, in_units);

      create_velocity_1(atom, vtot[0], vtot[1], vtot[2], comm_instance);
    }

    CkCallback cb(CkCallback(CkReductionTarget(Main, blocksCreated1), main_proxy));
    contribute(3*sizeof(double), vtot, CkReduction::set, cb);
  }

  void vtotReduced(double vxtot, double vytot, double vztot) {
    if (in_datafile.empty()) {
      create_velocity_2(in_t_request, atom, thermo, vxtot, vytot, vztot,
          comm_instance);
    }

    printConfig();

    contribute(CkCallback(CkReductionTarget(Main, blocksCreated2), main_proxy));
  }

  void printConfig() {
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
      CkPrintf("\t# Do safe exchange: %i\n", comm.do_safeexchange);
      CkPrintf("\t# Size of float: %i\n\n", (int) sizeof(MMD_float));
    }

  }

  ~Block() {}
};

#include "miniMD.def.h"
