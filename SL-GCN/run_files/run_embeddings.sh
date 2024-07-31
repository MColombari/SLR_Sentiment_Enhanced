#!/bin/bash

#SBATCH --account=cvcs2024
#SBATCH --job-name=testing
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00

#SBATCH --output="run_output/output_embedding.log"
#SBATCH --error="run_output/error_embedding.log"

# training 
python3 video2vec.py --config="config/sign/embeddings/embeddings_config.yaml"