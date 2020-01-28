#!/bin/bash
#BSUB -G asccasc
#BSUB -W 10
#BSUB -core_isolation 2
#BSUB -q pbatch
#BSUB -nnodes 256
#BSUB -J minimd-k-n1024

date

cd /g/g90/choi18/hpm/apps/miniMD/kokkos

ranks=1024

jsrun -n $ranks -a 1 -c 1 -g 1 -M "-gpu" ./miniMD-k -i in.lj.miniMD -nx 1024 -ny 512 -nz 512 > k-n"$ranks".out
