#!/bin/bash
#BSUB -P csc357
#BSUB -W 0:10
#BSUB -nnodes 1
#BSUB -J minimd-nsys-n4-c8

date

cd $MEMBERWORK/csc357/miniMD/charm

n_procs=4
n_chares=8
nx=128
ny=128
nz=128
n_iters=10
pemap="L0,4,84,88"
options="-gn 0"

jsrun -n$n_procs -a1 -c1 -g1 -K2 -r4 nvprof -o miniMD-n$n_procs-c$n_chares-nx$nx-ny$ny-nz$nz-p%q{OMPI_COMM_WORLD_RANK}.nvvp ./miniMD -c $n_chares -i ../inputs/in.lj.miniMD -nx $nx -ny $ny -nz $nz -n $n_iters $options +ppn 1 +pemap $pemap
