#!/bin/bash

# SLURM options:

#SBATCH --job-name=rew_tft    # Job name
#SBATCH --output=serial_test_%j.log   # Standard output and error log
#SBATCH --error=serial_test_error_%j.log


#SBATCH --partition=htc               # Partition choice (htc by default)

#SBATCH --ntasks=4                    # Run a single task
#SBATCH --mem=30000                    # Memory in MB per default
#SBATCH --time=2-00:00:00             # Max time limit = 7 days


#SBATCH --licenses=sps                # Declaration of storage and/or software resources

# Commands to be submitted:

eval "$(conda shell.bash hook)"
conda init

conda activate ltft_env

python neural_nets/wavelet_trafo.py
