#!/bin/bash
#BSUB -G asccasc
#BSUB -W 20
#BSUB -core_isolation 2
#BSUB -q pbatch
#BSUB -nnodes 3
#BSUB -J miniMD-charm-n3
#BSUB -o miniMD-charm-n3.%J

# These need to be changed between submissions
file=miniMD-e
n_nodes=3
n_procs=$((n_nodes * 4))
nx=384
ny=192
nz=192

# Function to display commands
exe() { echo "\$ $@" ; "$@" ; }

cd $HOME/miniMD/charm

ppn=1
pemap="L0,4,80,84"
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
    exe jsrun -n$n_procs -a1 -c$ppn -g1 -K2 -r4 ./$file -c $num_chares $options -nx $nx -ny $ny -nz $nz -n $n_iters +ppn $ppn +pemap $pemap
  done
done
