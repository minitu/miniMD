#!/bin/bash
#BSUB -P csc357
#BSUB -W 0:10
#BSUB -nnodes 2
#BSUB -J minimd-n8

date

cd $MEMBERWORK/csc357/hpm/apps/miniMD/kokkos

ranks=8

for iter in 1 2 3
do
  echo "Running iteration $iter"
  jsrun -n $ranks -a 1 -c 1 -g 1 -M "-gpu" ./miniMD-b -i in.lj.miniMD -nx 128 -ny 128 -nz 128 > n"$ranks"-"$iter".out
done
