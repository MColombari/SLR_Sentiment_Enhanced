#!/bin/bash

#SBATCH --account=cvcs2024
#SBATCH --job-name=bone_motion_gcn
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00

#SBATCH --output="run_output/output_train_bone_motion.log"
#SBATCH --error="run_output/error_train_bone_motion.log"

# training 
python3 main.py --config="config/sign/finetune/train_bone_motion.yaml"