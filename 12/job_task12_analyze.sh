#!/bin/bash
#BSUB -J task12_analyze
#BSUB -q hpc
#BSUB -W 10
#BSUB -n 1
#BSUB -R "rusage[mem=2048MB]"
#BSUB -o task12_analyze_%J.out
#BSUB -e task12_analyze_%J.err

module load python3

python3 analyze_results.py \
  --csv all_buildings_results.csv \
  --out-prefix task12
