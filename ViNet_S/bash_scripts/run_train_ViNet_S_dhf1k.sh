#!/bin/bash

source activate DL

echo "starting train.py"



dataset="DHF1K"
dataset_root_path="/mnt/Shared-Storage/rohit"
root_folder_path="/mnt/Shared-Storage/rohit/Home"
fold_lists_root_path="$root_folder_path/ICASSP_Saliency"

cd $root_folder_path/ICASSP_Saliency/ViNet_S
echo "starting train.py"

CUDA_VISIBLE_DEVICES=1, python3 train.py --videos_root_path $dataset_root_path/DHF1K \
--frames_path images \
--dataset $dataset \
--batch_size 8 \
--root_grouping True \
--grouped_conv True \
--model_save_path $root_folder_path/ICASSP_Saliency/ViNet_S/saved_models/${dataset}_vinet_s_rootgrouped_32_bs8_kld_cc.pt \
--no_epochs 120

echo "Done"