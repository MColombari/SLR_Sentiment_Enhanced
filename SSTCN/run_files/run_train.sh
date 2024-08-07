#!/bin/bash

#SBATCH --account=cvcs2024
#SBATCH --job-name=train_model
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00

#SBATCH --output="run_output/output_train.log"
#SBATCH --error="run_output/error_train.log"

####### training #############################
python train_parallel.py \
    --dataset_path /work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/SSTCN/data/train_features \
    --dataset_path_test /work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/SSTCN/data/test_features \
    --save_path /work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/SSTCN/model/model_checkpoints \
    --batch_size 60
