#!/bin/bash
#BSUB -P csc357
#BSUB -W 0:10
#BSUB -nnodes 1
#BSUB -J miniMD-charm-nsys

date

cd $MEMBERWORK/csc357/miniMD/charm

n_procs=6
n_chares=6
nx=192
ny=192
nz=192
n_iters=100
pemap="L0,4,8,84,88,92"
options="-gn 0"

jsrun -n$n_procs -a1 -c1 -g1 -K3 -r6 nsys profile -t cuda -s none -b none -f true -o miniMD-n$n_procs-c$n_chares-nx$nx-ny$ny-nz$nz-p%q{OMPI_COMM_WORLD_RANK} ./miniMD -c $n_chares -i ../inputs/in.lj.miniMD -nx $nx -ny $ny -nz $nz -n $n_iters $options +ppn 1 +pemap $pemap
