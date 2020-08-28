#!/bin/bash
#BSUB -W 10
#BSUB -P csc357
#BSUB -nnodes 1
#BSUB -J miniMD-mpi-n1

# These need to be changed between submissions
file=miniMD
n_nodes=1
n_procs=$((n_nodes * 6))
nx=192
ny=192
nz=192

# Function to display commands
exe() { echo "\$ $@" ; "$@" ; }

cd $HOME/work/miniMD/kokkos

n_iters=100
options="-i ../inputs/in.lj.miniMD -gn 0"

echo "# MiniMD (MPI + Kokkos) Performance Benchmarking"

for iter in 1 2 3 4 5
do
  echo -e "# Iteration $iter\n"
  exe jsrun -n$n_procs -a1 -c1 -g1 -K3 -r6 -M "-gpu" ./$file $options -nx $nx -ny $ny -nz $nz -n $n_iters
done
