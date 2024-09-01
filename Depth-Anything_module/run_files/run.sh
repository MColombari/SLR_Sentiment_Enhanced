#!/bin/bash

#SBATCH --account=cvcs2024
#SBATCH --job-name=train_NN_Ensemble
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

#SBATCH --output="run_output/output_train_NN_Ensemble.log"
#SBATCH --error="run_output/error_train_NN_Ensemble.log"

# training 
python3 /homes/mcolombari/SLR_Sentiment_Enhanced/Depth-Anything/run_video.py --encoder vits --video-path /work/cvcs2024/SLR_sentiment_enhanced/datasets/custom/raw_video/signer0_orientationLeft_sample0.mp4 --outdir /work/cvcs2024/SLR_sentiment_enhanced/Depth-Anything/Test