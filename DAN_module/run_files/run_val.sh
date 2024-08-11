#!/bin/bash

#SBATCH --account=cvcs2024
#SBATCH --job-name=DAN_val_genetarion
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

#SBATCH --output="run_output/output_val.log"
#SBATCH --error="run_output/error_val.log"

python gen_emotion.py \
    --video_folder /work/cvcs2024/SLR_sentiment_enhanced/datasets/WLASL/WLASL/start_kit/data/val \
    --npy_folder /work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/data-prepare/demo/val_npy \
    --out_file /work/cvcs2024/SLR_sentiment_enhanced/DAN/results/val.csv  \