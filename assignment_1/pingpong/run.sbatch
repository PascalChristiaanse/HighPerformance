#!/bin/sh
#
#SBATCH --job-name="hello_world"
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-EEMCS-Courses-WI4049TU

srun /home/pchristiaanse/HighPerformance/assignment_1/pingPong > /home/pchristiaanse/HighPerformance/assignment_1/output.log
  