#!/bin/bash -l

#SBATCH --job-name=test
#SBATCH -o /users/yx2022/sharedscratch/projects/hakcathon/logs/log_adamw.out
#SBATCH -e /users/yx2022/sharedscratch/projects/hakcathon/logs/log_adamw.err
#SBATCH --time=7-0:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
## Memory limit (in megabytes)
#SBATCH --mem=262144
#SBATCH -p gpu

flight start
flight env activate conda@myenv

srun python /users/yx2022/sharedscratch/projects/hakcathon/train_gsam_cross.py --total_epoch=200 --model='mobile' --optim='gsam' 