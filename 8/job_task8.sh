#!/bin/bash
#BSUB -J task8_cuda
#BSUB -q gpua10
#BSUB -W 60
#BSUB -n 1
#BSUB -R "rusage[mem=4096MB]"
#BSUB -o task8_%J.out
#BSUB -e task8_%J.err

module load python3
module load cuda

export LD_LIBRARY_PATH=${CUDA_HOME}/nvvm/lib64:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}

python3 -u solve_task8_cuda.py \
  --n-buildings 20 \
  --max-iter 20000 \
  --csv-out task8_results.csv
