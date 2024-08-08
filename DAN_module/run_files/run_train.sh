#!/bin/bash

#SBATCH --account=cvcs2024
#SBATCH --job-name=DAN_train_genetarion
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

#SBATCH --output="run_output/output_train.log"
#SBATCH --error="run_output/error_train.log"

python gen_emotion.py \
    --video_folder /work/cvcs2024/SLR_sentiment_enhanced/datasets/WLASL/WLASL/start_kit/data/train \
    --npy_folder /work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/data-prepare/demo/train_npy \
    --out_file /work/cvcs2024/SLR_sentiment_enhanced/DAN/results/train.csv  \