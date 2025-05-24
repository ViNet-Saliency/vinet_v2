#!/bin/bash

source activate DL

dataset="mvva"


echo "starting train.py"
cd ~/ICASSP_Saliency/ViNet_A




# CUDA_VISIBLE_DEVICES=3, python train.py --dataset $dataset \
#     --videos_root_path  /mnt/Shared-Storage/rohit/mvva_dataset \
#     --videos_frames_root_path /mnt/Shared-Storage/rohit/mvva_dataset/video_frames/mvva \
#     --gt_sal_maps_path /mnt/Shared-Storage/rohit/mvva_dataset/annotations/mvva \
#     --fixation_data_path /mnt/Shared-Storage/rohit/fixation_data_mvva \
#     --cc_coeff -1 \
#     --neck_name 'neck' \
#     --batch_size 2 \
#     --split 1 \
#     --len_snippet 64 \
#     --decoder_groups 32 \
#     --model_tag '' \
#     --no_epochs 2 \
#     --use_action_classification 0 \
#     --fold_lists_path /home/rohit/ICASSP_Saliency/fold_lists \
#     --model_save_root_path ~/ICASSP_Saliency/ViNet_A/saved_models \
#     --checkpoint_path '~/ICASSP_Saliency/ViNet_A/saved_models/ViNet_A_DHF1K_no_split_neck_32_kldiv_cc___6_0.pt'



dataset="av_datasets"

CUDA_VISIBLE_DEVICES=3, python train.py --dataset $dataset \
    --videos_root_path  /mnt/Shared-Storage/sid/datasets/Audio-Visual-SaliencyDatasets \
    --cc_coeff -1 \
    --neck_name 'neck' \
    --batch_size 2 \
    --split 1 \
    --len_snippet 64 \
    --decoder_groups 32 \
    --model_tag '' \
    --no_epochs 150 \
    --use_action_classification 0 \
    --fold_lists_path /home/rohit/ICASSP_Saliency/fold_lists \
    --model_save_root_path ~/ICASSP_Saliency/ViNet_A/saved_models \
    --checkpoint_path '~/ICASSP_Saliency/ViNet_A/saved_models/ViNet_A_DHF1K_no_split_neck_32_kldiv_cc___6_0.pt'












#     # --cc_coeff -1 \
#     # --neck_name 'neck' \
#     # --batch_size 6 \
#     # --split 1 \
#     # --window_length 32 \
#     # --len_snippet 64 \
#     # --decoder_groups 32 \
#     # --model_tag 'concat_av_64csz_32g_0.5s_removed_nans_noclip_seed2934_action_detection_run2' \
#     # --no_epochs 150 \
#     # --subset_type 'all' \
#     # --model_save_root_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/baseline/  \
#     # --use_action_classification 0 \
#     # --checkpoint_path /home/sid/SaliencyModel/testSal/DHF1K_baseline_32g_0.85294.pt















































# python train.py --dataset $dataset \
#     --videos_frames_root_path /home/sid/mvva_dataset/video_frames/mvva \
#     --videos_root_path /home/sid/mvva_dataset/mvva_raw_videos \
#     --gt_sal_maps_path /home/sid/mvva_dataset/annotations/mvva \
#     --fold_lists_path /home/sid/mvva_dataset/fold_lists \
#     --fixation_data_path /home/sid/mvva_dataset/fixation_data_mvva \
#     --audio_maps_path /home/sid/mvva_dataset/audio_spatial_maps \
#     --split 1 \
#     --video_frame_class_file_path /home/sid/mvva_dataset/video_frame_num_rois_class_dict.pickle \
#     --neck_name 'neck' \
#     --batch_size 2 \
#     --decoder_groups 32 \
#     --model_tag 'baseline_1gpus_4090_run1'


# python train.py --dataset $dataset \
#     --videos_root_path /home/sid/Audio-Visual-SaliencyDatasets \
#     --cc_coeff -1 \
#     --neck_name 'ASPP_neck' \
#     --batch_size 4 \
#     --window_length 32 \
#     --len_snippet 64 \
#     --decoder_groups 32 \
#     --model_tag 'ASPP_neck_1gpus_sid_extra_data_reloaded_best_0.5s_baseline_dhf1k_init_64clipsize_corrected_seed6161_run1' \
#     --no_epochs 150 \
#     --subset_type 'all' \
#     --use_image_saliency 0 \
#     --reload_data_every_epoch 1 \
#     --use_action_classification 0 \
#     --checkpoint_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/baseline_aspp/EEAA-B_random1_DHF1K_ASPP_neck_32_channel_shuffle_6_bs_ASPP_neck_1gpus_sid_full_data_best_0.5s_baseline_run1.pt
#--checkpoint_path /home/sid/SaliencyModel/testSal/DHF1K_baseline_32g_0.85294.pt
    # --checkpoint_path /home/sid/SaliencyModel/testSal/DHF1K_baseline_32g_0.85294.pt
    # --checkpoint_path /home/sid/SaliencyModel/testSal/DHF1K_baseline_32g_0.85294.pt

# python train.py --dataset $dataset \
#     --videos_root_path /home/sid/Audio-Visual-SaliencyDatasets \
#     --cc_coeff -1 \
#     --neck_name 'ensemble_neck' \
#     --batch_size 3 \
#     --window_length 32 \
#     --len_snippet 64 \
#     --decoder_groups 32 \
#     --model_tag 'ensemble_neck_1gpus_sid_ensemble_decoder_dissimilarity_image_saliency_neck_corrected_both_unfreezed_64_0.5stride_run1_modelC' \
#     --no_epochs 150 \
#     --modelA_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/ensemble_models/EEAA-B_random1_DHF1K_dissimilarity_neck_32_channel_shuffle_4_bs_dissimilarity_neck_1gpus_sid_human_best_0.5s_modelA_rerun1.pt \
#     --modelB_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/ensemble_models/EEAA-B_random1_DHF1K_image_saliency_neck_8_channel_shuffle_4_bs_image_saliency_neck_1gpus_sid_non_human_best_0.5s_modelB_rerun1.pt \
#     --subset_type 'all' \
#     --use_image_saliency 1 \
#     --checkpoint_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/EEAA-B_random1_DHF1K_ensemble_neck_32_channel_shuffle_6_bs_ensemble_neck_1gpus_sid_ensemble_decoder_image_saliency_neck_rerun1_modelC.pt \
#     --reload_data_every_epoch 1 \
#     --lr 1e-4

#'triplet_attention_neck2' #tripletAttention3D_neck
# module load u18/cuda/10.2
# module load u18/cudnn/7.6.5-cuda-10.2

# # echo "activating environment"
# # source activate DL

# # parent_directory="/ssd_scratch/cvit/jainsiddharth"

# # echo "copying dataset"
# # dataset_path="datasets"
# # dataset="DHF1K"
# # # dataset="UCF" 
# # mkdir -p "$parent_directory/$dataset_path/"
# # if [ ! -d "$parent_directory/$dataset_path/$dataset" ]; then
# #     rsync -r jainsiddharth@ada:/share3/dataset/Saliency/dhf1k/DHF1K /ssd_scratch/cvit/jainsiddharth/datasets/
# #     # rsync -r jainsiddharth@ada:/share3/dataset/Saliency/UCF /ssd_scratch/cvit/jainsiddharth/datasets/
# # fi

# # echo "copying checkpoint"
# # checkpoint_path="checkpoints"
# # checkpoint="AVA_SLOWFAST_R50_ACAR_HR2O.pth.tar"
# # mkdir -p "$parent_directory/$checkpoint_path/"
# # if [ ! -e "$parent_directory/$checkpoint_path/$checkpoint" ]; then
# #     rsync jainsiddharth@ada:/share3/jainsiddharth/checkpoints/AVA_SLOWFAST_R50_ACAR_HR2O.pth.tar /ssd_scratch/cvit/jainsiddharth/checkpoints/
# # fi

# # echo "starting train.py"
# # python train.py

# # source deactivate