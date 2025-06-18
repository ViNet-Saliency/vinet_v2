#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate DL

echo "starting train.py"
cd ~/ICASSP_Saliency/ViNet_A

dataset="mvva"
dataset_root_path="/mnt/Shared-Storage/rohit"
root_folder_path="/mnt/Shared-Storage/rohit/Home"
fold_lists_root_path="$root_folder_path/ICASSP_Saliency"

CUDA_VISIBLE_DEVICES=1, python train.py --dataset $dataset \
    --videos_frames_root_path $dataset_root_path/mvva_dataset/video_frames/mvva \
    --videos_root_path $dataset_root_path/mvva_dataset/mvva_raw_videos \
    --gt_sal_maps_path $dataset_root_path/mvva_dataset/annotations/mvva \
    --fold_lists_path $dataset_root_path/mvva_dataset/fold_lists \
    --fixation_data_path $dataset_root_path/fixation_data_mvva \
    --split 1 \
    --neck_name 'neck' \
    --batch_size 2 \
    --decoder_groups 32 \
    --use_action_classification 0 \
    --len_snippet 64 \
    --checkpoint_path '~/ICASSP_Saliency/ViNet_A/saved_models/ViNet_A_DHF1K_no_split_neck_32_kldiv_cc___6_0.pt' \
    --model_save_path $model_save_root_path/saved_models

