
#!/bin/bash
#BSUB -J mini_project_all_tests_64_floors
#BSUB -q hpc
#BSUB -W 120
#BSUB -R "rusage[mem=512MB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -n 16
#BSUB -o mini_project_all_tests_64_floors.out
#BSUB -e mini_project_all_tests_64_floors.err

lscpu
time python3 task_5.py 64 1
time python3 task_5.py 64 2
time python3 task_5.py 64 4
time python3 task_5.py 64 8
time python3 task_5.py 64 16

