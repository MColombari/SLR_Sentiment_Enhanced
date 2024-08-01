#!/bin/bash

#SBATCH --account=cvcs2024
#SBATCH --job-name=test_joint
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00

#SBATCH --output="run_output/output_test_joint.log"
#SBATCH --error="run_output/error_test_joint.log"

# training 
python3 main.py --config="config/sign/test/test_bone_motion.yaml"