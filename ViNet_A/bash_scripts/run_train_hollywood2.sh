#!/bin/bash
#SBATCH --job-name=acar_train_mvva
#SBATCH -A research
#SBATCH -c 28
#SBATCH --gres=gpu:3
#SBATCH -o /home/girmaji08/EEAA/SaliencyModel/logs/EEAA-B_split1_baseline_neck_3b_3gpus_run1.log
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END


# echo "activating environment"
# cd 
# source env_sal/bin/activate
# module load u18/cuda/11.6
# module load u18/cudnn/8.4.0-cuda-11.6
# module load u18/matlab/R2022a
# a=$USER

# parent_directory="/ssd_scratch/cvit/girmaji08"
# echo "copying dataset"
# dataset_path="Dataset/video_frames/"
source activate DL
################################################################################################################ 


dataset="Hollywood2"

echo "starting train.py"
cd ~/ICASSP_Saliency/ViNet_A

CUDA_VISIBLE_DEVICES=0, python train.py --dataset $dataset \
    --videos_root_path /mnt/Shared-Storage/sid/datasets/Hollywood2 \
    --cc_coeff -1 \
    --neck_name 'neck' \
    --batch_size 6 \
    --len_snippet 64 \
    --decoder_groups 32 \
    --model_tag '' \
    --no_epochs 150 \
    --use_action_classification 0 \
    --model_save_root_path ~/ICASSP_Saliency/ViNet_A/saved_models \
    --checkpoint_path '~/ICASSP_Saliency/ViNet_A/saved_models/ViNet_A_DHF1K_no_split_neck_32_kldiv_cc___6_0.pt'

# mkdir -p "$parent_directory/$dataset_path/"
# if [ ! -d "$parent_directory/$dataset_path/mvva" ]; then

#     mkdir -p $ssd_path

#     mkdir -p $ssd_path/Dataset/annotations
#     mkdir -p $ssd_path/Dataset/video_audio
#     mkdir -p $ssd_path/Dataset/video_frames

#     rsync -r girmaji08@ada.iiit.ac.in:/share3/girmaji08/mvva_dataset/annotations/mvva $ssd_path/Dataset/annotations

#     rsync -r girmaji08@ada.iiit.ac.in:/share3/girmaji08/mvva_dataset/video_audio/mvva $ssd_path/Dataset/video_audio

#     rsync -r girmaji08@ada.iiit.ac.in:/share3/girmaji08/mvva_dataset/video_frames/mvva $ssd_path/Dataset/video_frames

#     rsync -r girmaji08@ada.iiit.ac.in:/share3/girmaji08/mvva_dataset/fold_lists $ssd_path/Dataset/

#     rsync -r girmaji08@ada.iiit.ac.in:/share3/girmaji08/mvva_raw_videos $ssd_path/Dataset/

#     rsync -r girmaji08@ada.iiit.ac.in:/share3/girmaji08/fixation_data_mvva $ssd_path

# fi

# rsync -r girmaji08@ada.iiit.ac.in:/share3/girmaji08/fixation_data_mvva $ssd_path

# echo "starting train.py"
# cd /home/sid/SaliencyModel/EEAA/SaliencyModel



# python train.py --dataset $dataset \
#     --videos_root_path /home/sid/Hollywood2 \
#     --cc_coeff -1 \
#     --neck_name 'neck' \
#     --batch_size 6 \
#     --window_length 32 \
#     --len_snippet 64 \
#     --decoder_groups 32 \
#     --model_tag 'neck_6bs_64csz_newdhf1k_run1' \
#     --no_epochs 150 \
#     --checkpoint_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/baseline/EEAA-B_random1_DHF1K_neck_32_channel_shuffle_10_bs__neck_STAL_10bs_32g-0.822.pt \
#     --subset_type 'all' \
#     --model_save_root_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/baseline/ 

# python train.py --dataset $dataset \
#     --videos_root_path /home/sid/Hollywood2 \
#     --cc_coeff -1 \
#     --neck_name 'neck' \
#     --batch_size 3 \
#     --window_length 32 \
#     --len_snippet 32 \
#     --decoder_groups 16 \
#     --model_tag 'dhf1k_finetune_neck_1gpus_sid' \
#     --no_epochs 100 

# python train.py --dataset $dataset \
#     --videos_root_path /home/sid/Hollywood2 \
#     --cc_coeff -1 \
#     --neck_name 'ensemble_neck' \
#     --batch_size 3 \
#     --window_length 32 \
#     --len_snippet 64 \
#     --decoder_groups 32 \
#     --model_tag 'ensemble_neck_1gpus_sid_ensemble_decoder_dissimilarity_image_saliency_neck_human_unfreezed_64_1s_run1_modelC' \
#     --no_epochs 150 \
#     --modelA_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/EEAA-B_random1_DHF1K_dissimilarity_neck_32_channel_shuffle_4_bs_dissimilarity_neck_1gpus_sid_human_best_0.5s_modelA_rerun1.pt \
#     --modelB_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/EEAA-B_random1_DHF1K_image_saliency_neck_8_channel_shuffle_4_bs_image_saliency_neck_1gpus_sid_non_human_best_0.5s_modelB.pt \
#     --subset_type 'all' \
#     --use_image_saliency 1 \
#     --checkpoint_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/EEAA-B_random1_DHF1K_ensemble_neck_32_channel_shuffle_6_bs_ensemble_neck_1gpus_sid_ensemble_decoder_image_saliency_neck_rerun1_modelC.pt









# CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py --dataset mvva \
#     --videos_frames_root_path /ssd_scratch/cvit/girmaji08/Dataset/video_frames/mvva \
#     --videos_root_path /ssd_scratch/cvit/girmaji08/Dataset/mvva_raw_videos \
#     --gt_sal_maps_path /ssd_scratch/cvit/girmaji08/Dataset/annotations/mvva \
#     --fold_lists_path /ssd_scratch/cvit/girmaji08/Dataset/fold_lists \
#     --fixation_data_path /ssd_scratch/cvit/girmaji08/fixation_data_mvva \
#     --audio_maps_path /ssd_scratch/cvit/girmaji08/audio_spatial_maps \
#     --split 1 \
#     --video_frame_class_file_path /home/girmaji08/EEAA/SaliencyModel/video_frame_num_rois_class_dict.pickle \
#     --neck_name 'neck' \
#     --batch_size 1 \
#     --decoder_groups 32 \
#     --model_tag 'baseline_3gpus_run1'


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