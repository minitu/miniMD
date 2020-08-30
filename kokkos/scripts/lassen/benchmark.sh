#!/bin/bash
#BSUB -G asccasc
#BSUB -W 10
#BSUB -core_isolation 2
#BSUB -q pbatch
#BSUB -nnodes 3
#BSUB -J miniMD-mpi-hs-n3
#BSUB -o miniMD-mpi-hs-n3.%J

# These need to be changed between submissions
file=miniMD-hs
n_nodes=3
n_procs=$((n_nodes * 4))
nx=384
ny=192
nz=192

# Function to display commands
exe() { echo "\$ $@" ; "$@" ; }

cd $HOME/miniMD/kokkos

n_iters=100
options="-i ../inputs/in.lj.miniMD -gn 0"

echo "# MiniMD (MPI + Kokkos) Performance Benchmarking"

for iter in 1 2 3
do
  date
  echo -e "# Run $iter\n"
  exe jsrun -n$n_procs -a1 -c1 -g1 -K2 -r4 -M "-gpu" ./$file $options -nx $nx -ny $ny -nz $nz -n $n_iters
done
