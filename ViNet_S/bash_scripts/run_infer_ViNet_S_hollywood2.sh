#!/bin/bash

echo "activating environment"

source activate DL


dataset="Hollywood2"
dataset_root_path="/mnt/Shared-Storage/rohit"
root_folder_path="/mnt/Shared-Storage/rohit/Home"
fold_lists_root_path="$root_folder_path/ICASSP_Saliency"

cd $root_folder_path/ICASSP_Saliency/ViNet_S

CUDA_VISIBLE_DEVICES=1, python ViNet_S_inferences_metrics.py --dataset $dataset \
    --videos_root_path $dataset_root_path/Hollywood \
    --neck 'neck' \
    --decoder_groups 32 \
    --clip_size 32 \
    --root_grouping True \
    --grouped_conv True \
    --metrics_save_path $root_folder_path/ICASSP_Saliency/metrics_results \
    --save_path $root_folder_path/ICASSP_Saliency/inferences \
    --save_inferences 1 \
    --video_names_list 'actioncliptest00565' \
    --checkpoint_path $root_folder_path/ICASSP_Saliency/ViNet_S/saved_models/Hollywood2_vinet_s_rootgrouped_32_bs8_kld_cc.pt

