#!/bin/bash
#BSUB -G asccasc
#BSUB -W 10
#BSUB -core_isolation 2
#BSUB -q pbatch
#BSUB -nnodes 4
#BSUB -J minimd-d-n16

date

cd /g/g90/choi18/hpm/apps/miniMD/kokkos

ranks=16

echo "Generating DUMPI traces for $ranks ranks"

export LD_LIBRARY_PATH=$HOME/hpm/sst-dumpi/install/lib:$LD_LIBRARY_PATH
jsrun -n $ranks -a 1 -c 1 -g 1 -M "-gpu" ./miniMD-d -i in.lj.miniMD -nx 256 -ny 128 -nz 128 > d-n"$ranks".out
