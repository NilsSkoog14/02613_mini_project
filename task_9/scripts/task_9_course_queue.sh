#!/bin/bash
#BSUB -J task9_cupy_course_queue
#BSUB -q c02613
#BSUB -W 00:30
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=4096MB]"
#BSUB -R "select[gpu80gb]"
#BSUB -R "span[hosts=1]"
#BSUB -o task9_cupy_course_queue.out
#BSUB -e task9_cupy_course_queue.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

time python3 task_9.py \
  --n-buildings 20 \
  --max-iter 20000 \
  --csv-out task9_course_queue_results.csv
