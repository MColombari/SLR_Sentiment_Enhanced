#!/bin/bash

#SBATCH --account=cvcs2024
#SBATCH --job-name=test_bone_motion
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00

#SBATCH --output="run_output/output_test_bone_motion.log"
#SBATCH --error="run_output/error_test_bone_motion.log"

# training 
python3 main.py --config="config/sign/test/test_joint.yaml"