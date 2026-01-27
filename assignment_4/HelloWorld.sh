#!/bin/sh
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --account=Education-EEMCS-Courses-IN4049TU

srun HelloWorld
