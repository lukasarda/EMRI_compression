#!/bin/bash

# SLURM options:

#SBATCH --job-name=save_net    # Job name
#SBATCH --output=serial_test_%j.log   # Standard output and error log
#SBATCH --error=serial_test_error_%j.log




#SBATCH --ntasks=4                    # Run a single task
#SBATCH --mem=30000                    # Memory in MB per default
#SBATCH --time=2-00:00:00             # Max time limit = 7 days
#SBATCH --gres=gpu:v100:1              #specific request of a GPU type



#SBATCH --licenses=sps                # Declaration of storage and/or software resources

# Commands to be submitted:

#module add Programming_Languages/anaconda/3.10
eval "$(conda shell.bash hook)"
conda init

conda activate few_pytorch

python ./neural_nets/auto_enc_main.py

