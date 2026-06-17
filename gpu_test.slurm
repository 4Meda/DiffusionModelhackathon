#!/bin/bash -l

#SBATCH --job-name=test
#SBATCH -o /users/yx2022/sharedscratch/projects/hakcathon/logs/log_test.out
#SBATCH -e /users/yx2022/sharedscratch/projects/hakcathon/logs/log_test.err
#SBATCH --time=7-0:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
## Memory limit (in megabytes)
#SBATCH --mem=262144
#SBATCH -p gpu

flight start
flight env activate conda@myenv

srun python /users/yx2022/sharedscratch/projects/hakcathon/test.py --ckpt='model_180.pth'