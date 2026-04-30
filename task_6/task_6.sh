
#!/bin/bash
#BSUB -J task_6
#BSUB -q hpc
#BSUB -W 30
#BSUB -R "rusage[mem=512MB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -n 16
#BSUB -o task_6.out
#BSUB -e task_6.err

lscpu
time python3 task_5_and_6.py 64 1 1
time python3 task_5_and_6.py 64 2 1
time python3 task_5_and_6.py 64 4 1
time python3 task_5_and_6.py 64 8 1
time python3 task_5_and_6.py 64 16 1

