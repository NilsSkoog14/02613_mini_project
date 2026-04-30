#!/bin/bash
#BSUB -J task_9
#BSUB -q c02613
#BSUB -W 00:30
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=4096MB]"
#BSUB -R "select[gpu80gb]"
#BSUB -R "span[hosts=1]"
#BSUB -o task_9.out
#BSUB -e task_9.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

time python3 task_9.py \
  --n-buildings 20 \
  --max-iter 20000 \
  --csv-out task_9.csv
