
#!/bin/bash
#BSUB -J all_tests_dynamic
#BSUB -q hpc
#BSUB -W 30
#BSUB -R "rusage[mem=512MB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -n 16
#BSUB -o all_tests_dynamic.out
#BSUB -e all_tests_dynamic.err

lscpu
time python3 task_6.py 64 1
time python3 task_6.py 64 2
time python3 task_6.py 64 4
time python3 task_6.py 64 8
time python3 task_6.py 64 16

