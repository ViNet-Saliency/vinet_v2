
source activate DL

dataset="DHF1K"


echo "starting train.py"
cd ~/ICASSP_Saliency/ViNet_A

CUDA_VISIBLE_DEVICES=0, python train.py --dataset $dataset \
    --videos_root_path /mnt/Shared-Storage/sid/datasets/DHF1K \
    --cc_coeff -1 \
    --neck_name 'neck' \
    --batch_size 6 \
    --len_snippet 64 \
    --decoder_groups 32 \
    --no_epochs 150 \
    --use_action_classification 0 \
    --model_save_root_path ~/ICASSP_Saliency/ViNet_A/saved_models





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


# python train_img_sal_dist.py --dataset $dataset \
#     --videos_root_path /home/sid/DHF1K \
#     --cc_coeff -1 \
#     --neck_name 'image_saliency_distill_neck' \
#     --batch_size 5 \
#     --window_length 32 \
#     --len_snippet 64 \
#     --decoder_groups 32 \
#     --model_tag 'image_saliency_neck_action_classification_run1' \
#     --no_epochs 150 \
#     --subset_type 'all' \
#     --use_decoder_v2 0 \
#     --use_image_saliency 1 \
#     --reload_data_every_epoch 0 \
#     --use_action_classification 1 \
#     --model_save_root_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/baseline/


# python train.py --dataset $dataset \
#     --videos_root_path /home/sid/DHF1K \
#     --cc_coeff -1 \
#     --neck_name 'image_saliency_neck' \
#     --batch_size 6 \
#     --window_length 32 \
#     --len_snippet 64 \
#     --decoder_groups 32 \
#     --model_tag 'inverse_image_saliency_neck_action_classification_run1' \
#     --no_epochs 150 \
#     --subset_type 'all' \
#     --use_decoder_v2 0 \
#     --use_image_saliency 1 \
#     --reload_data_every_epoch 0 \
#     --use_action_classification 1 \
#     --model_save_root_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/baseline/

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



# python train.py --dataset $dataset \
#     --videos_root_path /home/sid/DHF1K \
#     --cc_coeff -1 \
#     --neck_name 'ASPP_multihead_neck' \
#     --batch_size 6 \
#     --window_length 32 \
#     --len_snippet 64 \
#     --decoder_groups 32 \
#     --model_tag 'ASPP_multihead_neck_1gpus_sid_full_data_best_0.5s_baseline_run1' \
#     --no_epochs 150 \
#     --subset_type 'all' \
#     --use_image_saliency 0 \
#     --reload_data_every_epoch 0 \
#     --use_action_classification 0


# python train.py --dataset $dataset \
#     --videos_root_path /home/sid/DHF1K \
#     --cc_coeff -1 \
#     --neck_name 'neck' \
#     --batch_size 6 \
#     --window_length 32 \
#     --len_snippet 64 \
#     --decoder_groups 32 \
#     --model_tag 'neck_1gpus_sid_fulldata_best_0.5s_seed0_action_classification_corrected_run1' \
#     --no_epochs 150 \
#     --subset_type 'all' \
#     --use_decoder_v2 0 \
#     --use_image_saliency 0 \
#     --reload_data_every_epoch 0 \
#     --use_action_classification 1 \
#     --model_save_root_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/baseline/
    # --checkpoint_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/baseline/EEAA-B_random1_DHF1K_neck_32_channel_shuffle_6_bs_neck_1gpus_sid_fulldata_best_0.5s_seed0_action_classification_run1.pt

# python train.py --dataset $dataset \
#     --videos_root_path /home/sid/DHF1K \
#     --cc_coeff -1 \
#     --neck_name 'dissimilarity_neck' \
#     --batch_size 4 \
#     --window_length 32 \
#     --len_snippet 64 \
#     --decoder_groups 32 \
#     --model_tag 'dissimilarity_neck_1gpus_sid_human_best_0.5s_modelA_rerun1' \
#     --no_epochs 150 \
#     --subset_type 'human' \
#     --use_image_saliency 0 \
#     --reload_data_every_epoch 0 \
#     --use_action_classification 0

# python train.py --dataset $dataset \
#     --videos_root_path /home/sid/DHF1K \
#     --cc_coeff -1 \
#     --neck_name 'ensemble_neck' \
#     --batch_size 6 \
#     --window_length 32 \
#     --len_snippet 64 \
#     --decoder_groups 32 \
#     --model_tag 'ensemble_neck_1gpus_sid_ensemble_decoder_dissimilarity_image_saliency_neck_AFB_rerun1_modelC' \
#     --no_epochs 150 \
#     --modelA_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/EEAA-B_random1_DHF1K_dissimilarity_neck_32_channel_shuffle_4_bs_dissimilarity_neck_1gpus_sid_human_best_0.5s_modelA_rerun1.pt \
#     --modelB_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/EEAA-B_random1_DHF1K_image_saliency_neck_8_channel_shuffle_4_bs_image_saliency_neck_1gpus_sid_non_human_best_0.5s_modelB.pt \
#     --subset_type 'all' \
#     --use_image_saliency 1


#--modelB_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/EEAA-B_random1_DHF1K_image_saliency_neck_8_channel_shuffle_4_bs_neck_1gpus_sid_non_human_image_saliency_modelB_best.pt \

#    --modelA_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/EEAA-B_random1_DHF1K_neck_32_channel_shuffle_4_bs_neck_1gpus_sid_human_modelA.pt \

#   --modelB_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/EEAA-B_random1_DHF1K_neck_16_channel_shuffle_4_bs_neck_1gpus_sid_non_human_modelB.pt \

#EEAA-B_random1_DHF1K_dissimilarity_neck_32_channel_shuffle_4_bs_dissimilarity_neck_1gpus_sid_human_best_0.5s_modelA_rerun1
#EEAA-B_random1_DHF1K_image_saliency_neck_8_channel_shuffle_4_bs_image_saliency_neck_1gpus_sid_non_human_best_0.5s_modelB_rerun1

#    --modelA_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/EEAA-B_random1_DHF1K_dissimilarity_neck_32_channel_shuffle_4_bs_dissimilarity_neck_1gpus_sid_human_modelA.pt \
#     --modelB_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/EEAA-B_random1_DHF1K_image_saliency_neck_8_channel_shuffle_4_bs_image_saliency_neck_1gpus_sid_non_human_best_0.5s_modelB.pt \
#    --modelA_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/EEAA-B_random1_DHF1K_dissimilarity_neck_32_channel_shuffle_4_bs_dissimilarity_neck_1gpus_sid_human_best_0.5s_modelA_rerun1.pt \


#### Best dissimilarity non-inverse image saliency
#   --modelA_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/EEAA-B_random1_DHF1K_dissimilarity_neck_32_channel_shuffle_4_bs_dissimilarity_neck_1gpus_sid_human_best_0.5s_modelA_rerun1.pt \
#     --modelB_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/EEAA-B_random1_DHF1K_image_saliency_neck_8_channel_shuffle_4_bs_image_saliency_neck_1gpus_sid_non_human_best_0.5s_modelB.pt \














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