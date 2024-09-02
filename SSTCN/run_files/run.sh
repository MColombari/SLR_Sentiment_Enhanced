#!/bin/bash

#SBATCH --account=cvcs2024
#SBATCH --job-name=SSTCN_Prepare
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00

#SBATCH --output="run_output/output.log"
#SBATCH --error="run_output/error.log"

cd data_process
python wholepose_features_extraction.py --video_path /work/cvcs2024/SLR_sentiment_enhanced/datasets/WLASL/WLASL/start_kit/data/train --feature_path /work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/SSTCN/data/train_features --istrain True
python wholepose_features_extraction.py --video_path /work/cvcs2024/SLR_sentiment_enhanced/datasets/WLASL/WLASL/start_kit/data/val --feature_path /work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/SSTCN/data/val_features

