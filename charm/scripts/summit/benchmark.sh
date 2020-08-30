#!/bin/bash
#BSUB -W 20
#BSUB -P csc357
#BSUB -nnodes 32
#BSUB -J miniMD-charm-n32

# These need to be changed between submissions
file=miniMD-e
n_nodes=64
n_procs=$((n_nodes * 6))
nx=768
ny=768
nz=768

# Function to display commands
exe() { echo "\$ $@" ; "$@" ; }

cd $HOME/work/miniMD/charm

ppn=1
pemap="L0,4,8,84,88,92"
n_iters=100
options="-i ../inputs/in.lj.miniMD -gn 0"

echo "# MiniMD (Charm++ + Kokkos) Performance Benchmarking"

for overdecomp in 1 2 4 8
do
  num_chares=$((n_procs * overdecomp))

  echo -e "# ODF-$overdecomp\n"
  for iter in 1 2 3
  do
    date
    echo -e "# Run $iter\n"
    exe jsrun -n$n_procs -a1 -c$ppn -g1 -K3 -r6 ./$file -c $num_chares $options -nx $nx -ny $ny -nz $nz -n $n_iters +ppn $ppn +pemap $pemap
  done
done
