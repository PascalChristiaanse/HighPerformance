#!/bin/sh
#
#SBATCH --job-name="Parallel Poisson Solver"
#SBATCH --partition=compute
#SBATCH --time=00:01:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-EEMCS-Courses-WI4049TU

srun /home/pchristiaanse/HighPerformance/assignment_2/PAR_Poisson > /home/pchristiaanse/HighPerformance/assignment_2/output.log
  