#!/bin/bash



dataset="mvva"
dataset_root_path="/mnt/Shared-Storage/rohit"
root_folder_path="/mnt/Shared-Storage/rohit/Home"
fold_lists_root_path="$root_folder_path/ICASSP_Saliency"

cd $root_folder_path/ICASSP_Saliency/ViNet_A


CUDA_VISIBLE_DEVICES=1, python ViNet_A_inferences_metrics.py --dataset $dataset \
    --videos_frames_root_path $dataset_root_path/mvva_dataset/video_frames/mvva \
    --videos_root_path $dataset_root_path/mvva_dataset/mvva_raw_videos \
    --gt_sal_maps_path $dataset_root_path/mvva_dataset/annotations/mvva \
    --fold_lists_path $dataset_root_path/mvva_dataset/fold_lists \
    --fixation_data_path $dataset_root_path/fixation_data_mvva \
	--split 1 \
	--neck_name 'neck' \
	--decoder_groups 32 \
	--len_snippet 64 \
	--save_path $root_folder_path/ICASSP_Saliency/inferences \
	--save_inferences 1 \
    --video_names_list '001' \
	--metrics_save_path $root_folder_path/ICASSP_Saliency/metrics_results \
	--checkpoint_path $root_folder_path/ICASSP_Saliency/ViNet_A/saved_models/ViNet_A_mvva_1_neck_32_kldiv_cc___2_0.pt