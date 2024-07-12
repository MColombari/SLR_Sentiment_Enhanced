#!/bin/bash

#SBATCH --account=cvcs2024
#SBATCH --job-name=test_repo
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --time=1:00:00

#SBATCH --output="run_output/output.log"
#SBATCH --error="run_output/error.log"

cd data_process
python wholepose_features_extraction.py --video_path /work/cvcs2024/SLR_sentiment_enhanced/datasets/WLASL/WLASL/start_kit/data/train --feature_path ../data/train_features --istrain True
python wholepose_features_extraction.py --video_path /work/cvcs2024/SLR_sentiment_enhanced/datasets/WLASL/WLASL/start_kit/data/val --feature_path ../data/train_features
    #cd ..
# if you want to delete videos, un common the following command
#rm -rf train_videos
#rm -rf val_videos

####### training #############################
    #python train_parallel.py --batch_size 160
###### testing ###########################
    #python test.py --checkpoint_model /work/cvcs2024/SLR_sentiment_enhanced/model_weights/SSTCN/T_Pose_model_final.pth
#python test.py --checkpoint_model model_checkpoints/your model
