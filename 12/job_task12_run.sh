#!/bin/bash
#BSUB -J task12_run
#BSUB -q hpc
#BSUB -W 120
#BSUB -n 8
#BSUB -R "span[hosts=1] rusage[mem=2048MB]"
#BSUB -o task12_run_%J.out
#BSUB -e task12_run_%J.err

module load python3

PYTHONUNBUFFERED=1 python3 -u run_all_buildings.py \
  --n-workers 8 \
  --max-iter 20000 \
  --atol 1e-4 \
  --csv-out all_buildings_results.csv
