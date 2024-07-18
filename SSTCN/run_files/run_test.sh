#!/bin/bash

#SBATCH --account=cvcs2024
#SBATCH --job-name=test_model
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00

#SBATCH --output="run_output/output_test.log"
#SBATCH --error="run_output/error_test.log"

###### testing ###########################
python test.py \
    --dataset_path /work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/SSTCN/data/test_feature \
    --checkpoint_model /work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/SSTCN/model_checkpoints/T_Pose_model_100_32.463768115942024.pth \
    --test_labels ./test_labels_WLASL.pkl
