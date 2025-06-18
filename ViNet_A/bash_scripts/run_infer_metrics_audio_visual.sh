#!/bin/bash



dataset="ETMD_av"
dataset_root_path="/mnt/Shared-Storage/rohit"
root_folder_path="/mnt/Shared-Storage/rohit/Home"
fold_lists_root_path="$root_folder_path/ICASSP_Saliency"

cd $root_folder_path/ICASSP_Saliency/ViNet_A


CUDA_VISIBLE_DEVICES=1, python ViNet_A_inferences_metrics.py --dataset $dataset \
	--videos_root_path  $dataset_root_path/Audio-Visual-SaliencyDatasets \
	--fold_lists_path $dataset_root_path/Audio-Visual-SaliencyDatasets/fold_lists \
	--split 1 \
	--neck_name 'neck' \
	--decoder_groups 32 \
	--len_snippet 64 \
	--save_path $root_folder_path/ICASSP_Saliency/inferences \
	--save_inferences 1 \
    --video_names_list 'CHI_1_color' \
	--metrics_save_path $root_folder_path/ICASSP_Saliency/metrics_results \
	--checkpoint_path $root_folder_path/ICASSP_Saliency/ViNet_A/saved_models/ViNet_A_ETMD_av_1_neck_32_kldiv_cc___2_0.pt