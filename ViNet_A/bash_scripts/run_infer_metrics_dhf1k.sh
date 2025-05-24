#!/bin/bash

echo "activating environment"
cd
source activate DL

dataset="DHF1K"

echo "starting ViNet_A_inferences_metrics.py"
cd ~/ICASSP_Saliency/ViNet_A


CUDA_VISIBLE_DEVICES=0, python ViNet_A_inferences_metrics.py --dataset $dataset \
    --videos_root_path /mnt/Shared-Storage/sid/datasets/DHF1K \
    --neck 'neck' \
    --batch_size 1 \
    --decoder_groups 32 \
    --len_snippet 64 \
    --metrics_save_path '~/ICASSP_Saliency/metrics_results' \
    --save_path '~/ICASSP_Saliency/inferences' \
    --save_inferences 0 \
    --video_names_list '0665' \
    --checkpoint_path '~/ICASSP_Saliency/ViNet_A/saved_models/ViNet_A_DHF1K_no_split_neck_32_kldiv_cc___6_0.pt'




    # --checkpoint_path '/home/sid/fantastic_gains/saved_models/KD/ViNet-KD_SPATIAL_STAL_to_AC_stal_DHF1K_random1_neck_32_kldivcc_4bs_seed0.pt'

   
    # --checkpoint_path '/home/sid/fantastic_gains/saved_models/KD/ViNet-KD_STAL_to_AC_extra_data_alllosses_stal_DHF1K_random1_neck_32_kldivcc_4bs_seed0.pt'
 
     	# --checkpoint_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/baseline/DHF1K_STAL_32g_6bs_0.85294.pt' \

    # --modelA_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/EEAA-B_random1_DHF1K_dissimilarity_neck_32_channel_shuffle_4_bs_dissimilarity_neck_1gpus_sid_human_best_0.5s_modelA_rerun1.pt \
    # --modelB_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/EEAA-B_random1_DHF1K_image_saliency_neck_8_channel_shuffle_4_bs_image_saliency_neck_1gpus_sid_non_human_best_0.5s_modelB.pt \
    # --subset_type 'all' \
    # --use_image_saliency 1



    #EEAA-B_random1_DHF1K_ensemble_neck_32_channel_shuffle_6_bs_ensemble_neck_1gpus_sid_ensemble_decoder_image_saliency_modelC


    #EEAA-B_random1_DHF1K_ensemble_neck_32_channel_shuffle_6_bs_ensemble_neck_1gpus_sid_ensemble_modelC

    # --modelA_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/EEAA-B_random1_DHF1K_neck_32_channel_shuffle_4_bs_neck_1gpus_sid_human_modelA.pt \
    # --modelB_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/EEAA-B_random1_DHF1K_image_saliency_neck_8_channel_shuffle_4_bs_neck_1gpus_sid_non_human_image_saliency_modelB_best.pt \


# The best model ensemble model till now is EEAA-B_random1_DHF1K_ensemble_neck_32_channel_shuffle_6_bs_ensemble_neck_1gpus_sid_ensemble_decoder_image_saliency_neck_rerun1_modelC.pt