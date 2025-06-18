#!/bin/bash

echo "activating environment"

source activate DL


dataset="UCF"
dataset_root_path="/mnt/Shared-Storage/rohit"
root_folder_path="/mnt/Shared-Storage/rohit/Home"
fold_lists_root_path="$root_folder_path/ICASSP_Saliency"

cd $root_folder_path/ICASSP_Saliency/ViNet_A

CUDA_VISIBLE_DEVICES=1, python ViNet_A_inferences_metrics.py --dataset $dataset \
    --videos_root_path $dataset_root_path/UCF \
    --neck 'neck' \
    --decoder_groups 32 \
    --len_snippet 64 \
    --metrics_save_path $root_folder_path/ICASSP_Saliency/metrics_results \
    --save_path $root_folder_path/ICASSP_Saliency/inferences \
    --save_inferences 1 \
    --video_names_list 'Run-Side-001' \
    --checkpoint_path $root_folder_path/ICASSP_Saliency/ViNet_A/saved_models/ViNet_A_UCF_no_split_neck_32_kldiv_cc___6_0.pt

























# echo "activating environment"
# cd 

# source activate DL

# ################################################################################################################ 


# dataset="UCF"



# echo "starting inferences_metrics.py"
# cd ~/SaliencyModel/EEAA/SaliencyModel/

# python inferences_metrics.py --dataset $dataset \
# 	--videos_root_path /home/sid/UCF \
# 	--neck_name 'neck' \
# 	--batch_size 1 \
# 	--decoder_groups 32 \
#     --window_length 32 \
# 	--len_snippet 64 \
# 	--checkpoint_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/baseline/ViNet_A_UCF_no_split_neck_32_kldiv_cc___6_0.pt' \
# 	--metrics_save_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/metrics_results'