#!/bin/bash

#SBATCH --account=cvcs2024
#SBATCH --job-name=test_Conv
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00

#SBATCH --output="run_output/rgb_test_output.log"
#SBATCH --error="run_output/rgb_test_error.log"
python3 Sign_Isolated_Conv3D_clip_test.py