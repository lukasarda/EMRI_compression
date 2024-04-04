#!/bin/bash

# SLURM options:

#SBATCH --job-name=30000    # Job name
#SBATCH --output=serial_test_%j.log   # Standard output and error log
#SBATCH --error=serial_test_error_%j.log




#SBATCH --ntasks=4                    # Run a single task
#SBATCH --mem=30000                    # Memory in MB per default
#SBATCH --time=1-00:00:00             # Max time limit = 7 days
#SBATCH --gres=gpu:v100:1              #specific request of a GPU type


#SBATCH --mail-user=None   # Where to send mail
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)

#SBATCH --licenses=sps                # Declaration of storage and/or software resources

# Commands to be submitted:

#module add Programming_Languages/anaconda/3.10
eval "$(conda shell.bash hook)"
conda init

conda activate few_pytorch

#python generate_wf_td.py
#python generate_wf_td.py

#python main.py

python analysis_script.py
#python cuml_inc_PCA.py

#python npz_to_pt.py
