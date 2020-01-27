#!/bin/bash
#BSUB -G asccasc
#BSUB -W 10
#BSUB -core_isolation 2
#BSUB -q pbatch
#BSUB -nnodes 4
#BSUB -J minimd-n16

date

cd /g/g90/choi18/hpm/apps/miniMD/kokkos

ranks=16

for iter in 1 2 3
do
  echo "Running iteration $iter"
  jsrun -n $ranks -a 1 -c 1 -g 1 -M "-gpu" ./miniMD-b -i in.lj.miniMD -nx 256 -ny 128 -nz 128 > n"$ranks"-"$iter".out
done
