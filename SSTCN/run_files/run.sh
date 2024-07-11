#!/bin/bash

#SBATCH --account=cvcs2024
#SBATCH --job-name=test_repo
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --time=1:00:00

#SBATCH --output="run_output/output.log"
#SBATCH --error="run_output/error.log"

python3 main_process.sh --config="config/sign/test/test_bone_motion.yaml"