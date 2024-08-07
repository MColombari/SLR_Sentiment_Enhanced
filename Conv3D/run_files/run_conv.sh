#!/bin/bash

#SBATCH --account=cvcs2024
#SBATCH --job-name=test_repo
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --time=18:00:00

#SBATCH --output="run_output/rgb_output.log"
#SBATCH --error="run_output/rgb_error.log"
python3 Sign_Isolated_Conv3D_clip.py