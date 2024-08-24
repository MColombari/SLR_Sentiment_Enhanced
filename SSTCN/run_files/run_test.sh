#!/bin/bash

#SBATCH --account=cvcs2024
#SBATCH --job-name=test_model
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --time=6:00:00

#SBATCH --output="run_output/output_test.log"
#SBATCH --error="run_output/error_test.log"

###### testing ###########################
python test.py \
    --dataset_path /work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/SSTCN/data/val_features \
    --checkpoint_model /work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/SSTCN/model/model_checkpoints/T_Pose_model_298_32.61904761904762.pth \
    --test_labels ./val_labels_WLASL.pkl \
    --out_file /work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/SSTCN/val_out.pkl
