#!/bin/bash
#BSUB -J task7_numba_cpu
#BSUB -q hpc
#BSUB -W 60
#BSUB -n 1
#BSUB -R "rusage[mem=4096MB]"
#BSUB -o task7_%J.out
#BSUB -e task7_%J.err

module load python3

python3 solve_task7.py \
  --n-buildings 20 \
  --max-iter 20000 \
  --atol 1e-4 \
  --csv-out task7_results.csv
