#!/bin/bash

#SBATCH -J Sindex          # Job name
#SBATCH -A m2_jgu-paa      # Account name
#SBATCH -p m2_gpu          # Partition name
#SBATCH -t 300             # Time in minutes
#SBATCH -n 1               # Number of tasks
#SBATCH -c 8               # Number of CPUs
#SBATCH --mem=10G          # Memory in per node
#SBATCH --gres=gpu:1       # Reserve GPUs

./benchmark