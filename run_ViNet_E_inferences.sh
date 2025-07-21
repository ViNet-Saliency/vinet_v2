#!/bin/bash


echo "activating environment"

source activate DL


dataset="DHF1K"

dataset_root_path="/mnt/Shared-Storage/rohit"
root_folder_path="/mnt/Shared-Storage/rohit/Home"
fold_lists_root_path="$root_folder_path/ICASSP_Saliency"

cd $root_folder_path/ICASSP_Saliency/


CUDA_VISIBLE_DEVICES=1, python ViNet_E_inferences_metrics.py --dataset $dataset \
    --videos_root_path $dataset_root_path/DHF1K \
    --neck 'neck' \
    --decoder_groups 32 \
    --len_snippet 64 \
    --root_grouping True \
    --grouped_conv True \
    --metrics_save_path $root_folder_path/ICASSP_Saliency/metrics_results \
    --save_path $root_folder_path/ICASSP_Saliency/inferences \
    --save_inferences 1 \
    --video_names_list '0665' \
    --checkpoint_path_1 $root_folder_path/ICASSP_Saliency/ViNet_A/saved_models/ViNet_A_DHF1K_no_split_neck_32_kldiv_cc___6_0.pt \
	--checkpoint_path_2 $root_folder_path/ICASSP_Saliency/ViNet_S/saved_models/DHF1K_vinet_s_rootgrouped_32_bs8_kld_cc.pt \


dataset="ETMD_av"
dataset_root_path="/mnt/Shared-Storage/rohit"
root_folder_path="/mnt/Shared-Storage/rohit/Home"
fold_lists_root_path="$root_folder_path/ICASSP_Saliency"

cd $root_folder_path/ICASSP_Saliency/

CUDA_VISIBLE_DEVICES=1, python ViNet_E_inferences_metrics.py --dataset $dataset \
	--videos_root_path  $dataset_root_path/Audio-Visual-SaliencyDatasets \
	--fold_lists_path $dataset_root_path/Audio-Visual-SaliencyDatasets/fold_lists \
    --neck 'neck' \
    --decoder_groups 32 \
    --len_snippet 64 \
    --root_grouping True \
    --grouped_conv True \
    --split 1 \
    --metrics_save_path $root_folder_path/ICASSP_Saliency/metrics_results \
    --save_path $root_folder_path/ICASSP_Saliency/inferences \
    --save_inferences 1 \
    --video_names_list 'CHI_1_color' \
    --checkpoint_path_2 $root_folder_path/ICASSP_Saliency/ViNet_S/saved_models/ETMD_av_vinet_s_rootgrouped_32_bs8_kld_cc.pt \
    --checkpoint_path_1 $root_folder_path/ICASSP_Saliency/ViNet_A/saved_models/ViNet_A_ETMD_av_1_neck_32_kldiv_cc___2_0.pt



dataset="Hollywood2"
dataset_root_path="/mnt/Shared-Storage/rohit"
root_folder_path="/mnt/Shared-Storage/rohit/Home"
fold_lists_root_path="$root_folder_path/ICASSP_Saliency"

cd $root_folder_path/ICASSP_Saliency/

CUDA_VISIBLE_DEVICES=1, python ViNet_E_inferences_metrics.py --dataset $dataset \
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
    --checkpoint_path_2 $root_folder_path/ICASSP_Saliency/ViNet_S/saved_models/Hollywood2_vinet_s_rootgrouped_32_bs8_kld_cc.pt \
    --checkpoint_path_1 $root_folder_path/ICASSP_Saliency/ViNet_A/saved_models/ViNet_A_Hollywood2_no_split_neck_32_kldiv_cc___6_0.pt




dataset="UCF"
dataset_root_path="/mnt/Shared-Storage/rohit"
root_folder_path="/mnt/Shared-Storage/rohit/Home"
fold_lists_root_path="$root_folder_path/ICASSP_Saliency"

cd $root_folder_path/ICASSP_Saliency/

CUDA_VISIBLE_DEVICES=1, python ViNet_E_inferences_metrics.py --dataset $dataset \
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
    --checkpoint_path_2 $root_folder_path/ICASSP_Saliency/ViNet_S/saved_models/UCF_vinet_s_rootgrouped_32_bs8_kld_cc.pt \
    --checkpoint_path_1 $root_folder_path/ICASSP_Saliency/ViNet_A/saved_models/ViNet_A_UCF_no_split_neck_32_kldiv_cc___6_0.pt




dataset="mvva"
dataset_root_path="/mnt/Shared-Storage/rohit"
root_folder_path="/mnt/Shared-Storage/rohit/Home"
fold_lists_root_path="$root_folder_path/ICASSP_Saliency"

cd $root_folder_path/ICASSP_Saliency/

CUDA_VISIBLE_DEVICES=1, python ViNet_E_inferences_metrics.py --dataset $dataset \
    --videos_frames_root_path $dataset_root_path/mvva_dataset/video_frames/mvva \
    --videos_root_path $dataset_root_path/mvva_dataset/mvva_raw_videos \
    --gt_sal_maps_path $dataset_root_path/mvva_dataset/annotations/mvva \
    --fold_lists_path $dataset_root_path/mvva_dataset/fold_lists \
    --fixation_data_path $dataset_root_path/fixation_data_mvva \
    --neck 'neck' \
    --decoder_groups 32 \
    --len_snippet 32 \
    --root_grouping True \
    --grouped_conv True \
    --split 1 \
    --metrics_save_path $root_folder_path/ICASSP_Saliency/metrics_results \
    --save_path $root_folder_path/ICASSP_Saliency/inferences \
    --save_inferences 1 \
    --video_names_list '001' \
    --checkpoint_path_2 $root_folder_path/ICASSP_Saliency/ViNet_S/saved_models/mvva_vinet_s_rootgrouped_32_bs8_kld_cc.pt \
    --checkpoint_path_1 $root_folder_path/ICASSP_Saliency/ViNet_A/saved_models/ViNet_A_mvva_1_neck_32_kldiv_cc___2_0.pt

