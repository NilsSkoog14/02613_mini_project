¨
#!/bin/bash
#BSUB -J task_5
#BSUB -q hpc
#BSUB -W 120
#BSUB -n 16
#BSUB -R "rusage[mem=512MB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -o task_5.out
#BSUB -e task_5.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

lscpu
time python3 task_5_and_6.py 64 1
time python3 task_5_and_6.py 64 2
time python3 task_5_and_6.py 64 4
time python3 task_5_and_6.py 64 8
time python3 task_5_and_6.py 64 16

