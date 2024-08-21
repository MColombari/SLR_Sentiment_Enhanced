#!/bin/bash

#SBATCH --account=cvcs2024
#SBATCH --job-name=train_NN_Ensemble
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

#SBATCH --output="run_output/output_test_rotated_joint.log"
#SBATCH --error="run_output/error_test_rotated_joint.log"

# training 
python3 ensemble_NN.py