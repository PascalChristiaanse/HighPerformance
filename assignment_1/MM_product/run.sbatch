#!/bin/sh
#
#SBATCH --job-name="MM_product"
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-EEMCS-Courses-WI4049TU

srun /home/pchristiaanse/HighPerformance/assignment_1/MM_product/MM-product > /home/pchristiaanse/HighPerformance/assignment_1/MM_product/MM-product_output.log
  