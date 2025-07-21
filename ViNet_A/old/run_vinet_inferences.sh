#!/bin/bash

echo "activating environment"
cd
source activate DL



echo "starting vinet_inferences.py"
cd ~/SaliencyModel/EEAA/SaliencyModel/


dataset="DHF1K"
cd ~/SaliencyModel/EEAA/SaliencyModel/
python vinet_inferences.py --dataset $dataset \
	--videos_root_path /home/sid/DHF1K \
	--train_path_data /home/sid/$dataset/annotation \
	--val_path_data /home/sid/$dataset/val \
	--split 1 \
	--video_frame_class_file_path /home/sid/mvva_dataset/video_frame_num_rois_class_dict.pickle \
	--neck_name 'neck' \
	--batch_size 1 \
	--decoder_groups 32 \
	--len_snippet 32 \
	--save_path '/home/sid/' \
	--checkpoint_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/vinet--/vinet_rootgrouped_32_dhf1k.pt' \
	--metrics_save_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/metrics_results' \
	--save_inferences 1 \
	--video_names_list '0665'



# dataset="Coutrot_db2"
# python vinet_inferences.py --dataset $dataset \
# 	--videos_frames_root_path /home/sid/Audio-Visual-SaliencyDatasets/video_frames/$dataset \
# 	--gt_sal_maps_path /home/sid/Audio-Visual-SaliencyDatasets/annotations/$dataset \
# 	--fold_lists_path /home/sid/Audio-Visual-SaliencyDatasets/fold_lists \
# 	--fixation_data_path /home/sid/mvva_dataset/fixation_data_mvva \
# 	--audio_maps_path /ssd_scratch/cvit/girmaji08/audio_spatial_maps \
# 	--split 1 \
# 	--video_frame_class_file_path /home/sid/mvva_dataset/video_frame_num_rois_class_dict.pickle \
# 	--neck_name 'neck' \
# 	--batch_size 1 \
# 	--decoder_groups 32 \
# 	--len_snippet 32 \
# 	--save_path '/home/sid/' \
# 	--checkpoint_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/vinet--/vinet_rootgrouped_32_coutrot2_split1_bs4_kldv_cc.pt' \
# 	--metrics_save_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/metrics_results'
# 	--save_inferences 1 \
# 	--video_names_list '0700'


# dataset="mvva"
# python vinet_inferences.py --dataset $dataset \
# 	--videos_frames_root_path /home/sid/mvva_dataset/video_frames/mvva \
# 	--videos_root_path /home/sid/mvva_dataset/mvva_raw_videos \
# 	--gt_sal_maps_path /home/sid/mvva_dataset/annotations/mvva \
# 	--fold_lists_path /home/sid/mvva_dataset/fold_lists \
# 	--fixation_data_path /home/sid/mvva_dataset/fixation_data_mvva \
# 	--audio_maps_path /ssd_scratch/cvit/girmaji08/audio_spatial_maps \
# 	--split 1 \
# 	--video_frame_class_file_path /home/sid/mvva_dataset/video_frame_num_rois_class_dict.pickle \
# 	--neck_name 'neck' \
# 	--batch_size 1 \
# 	--decoder_groups 32 \
# 	--len_snippet 32 \
# 	--save_path '/home/sid/' \
# 	--checkpoint_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/vinet--/ViNet-S_mvva_8bs_32g.pt' \
# 	--metrics_save_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/metrics_results'

# dataset="DIEM"
# python vinet_inferences.py --dataset $dataset \
# 	--videos_frames_root_path /home/sid/Audio-Visual-SaliencyDatasets/video_frames/$dataset \
# 	--gt_sal_maps_path /home/sid/Audio-Visual-SaliencyDatasets/annotations/$dataset \
# 	--fold_lists_path /home/sid/Audio-Visual-SaliencyDatasets/fold_lists \
# 	--fixation_data_path /home/sid/mvva_dataset/fixation_data_mvva \
# 	--audio_maps_path /ssd_scratch/cvit/girmaji08/audio_spatial_maps \
# 	--split 1 \
# 	--video_frame_class_file_path /home/sid/mvva_dataset/video_frame_num_rois_class_dict.pickle \
# 	--neck_name 'neck' \
# 	--batch_size 1 \
# 	--decoder_groups 32 \
# 	--len_snippet 32 \
# 	--save_path '/home/sid/' \
# 	--checkpoint_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/vinet--/vinet_rootgrouped_32_diem_bs4_kldiv_cc.pt' \
# 	--metrics_save_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/metrics_results'



# dataset="ETMD_av"
# python vinet_inferences.py --dataset $dataset \
# 	--videos_frames_root_path /home/sid/Audio-Visual-SaliencyDatasets/video_frames/$dataset \
# 	--gt_sal_maps_path /home/sid/Audio-Visual-SaliencyDatasets/annotations/$dataset \
# 	--fold_lists_path /home/sid/Audio-Visual-SaliencyDatasets/fold_lists \
# 	--fixation_data_path /home/sid/mvva_dataset/fixation_data_mvva \
# 	--audio_maps_path /ssd_scratch/cvit/girmaji08/audio_spatial_maps \
# 	--split 1 \
# 	--video_frame_class_file_path /home/sid/mvva_dataset/video_frame_num_rois_class_dict.pickle \
# 	--neck_name 'neck' \
# 	--batch_size 1 \
# 	--decoder_groups 32 \
# 	--len_snippet 32 \
# 	--save_path '/home/sid/' \
# 	--checkpoint_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/vinet--/vinet_rootgrouped_32_etmd_split1_bs4.pt' \
# 	--metrics_save_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/metrics_results'

# python vinet_inferences.py --dataset $dataset \
# 	--videos_frames_root_path /home/sid/Audio-Visual-SaliencyDatasets/video_frames/$dataset \
# 	--gt_sal_maps_path /home/sid/Audio-Visual-SaliencyDatasets/annotations/$dataset \
# 	--fold_lists_path /home/sid/Audio-Visual-SaliencyDatasets/fold_lists \
# 	--fixation_data_path /home/sid/mvva_dataset/fixation_data_mvva \
# 	--audio_maps_path /ssd_scratch/cvit/girmaji08/audio_spatial_maps \
# 	--split 2 \
# 	--video_frame_class_file_path /home/sid/mvva_dataset/video_frame_num_rois_class_dict.pickle \
# 	--neck_name 'neck' \
# 	--batch_size 1 \
# 	--decoder_groups 32 \
# 	--len_snippet 32 \
# 	--save_path '/home/sid/' \
# 	--checkpoint_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/vinet--/vinet_rootgrouped_32_etmd_split2_bs4.pt' \
# 	--metrics_save_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/metrics_results'

# python vinet_inferences.py --dataset $dataset \
# 	--videos_frames_root_path /home/sid/Audio-Visual-SaliencyDatasets/video_frames/$dataset \
# 	--gt_sal_maps_path /home/sid/Audio-Visual-SaliencyDatasets/annotations/$dataset \
# 	--fold_lists_path /home/sid/Audio-Visual-SaliencyDatasets/fold_lists \
# 	--fixation_data_path /home/sid/mvva_dataset/fixation_data_mvva \
# 	--audio_maps_path /ssd_scratch/cvit/girmaji08/audio_spatial_maps \
# 	--split 3 \
# 	--video_frame_class_file_path /home/sid/mvva_dataset/video_frame_num_rois_class_dict.pickle \
# 	--neck_name 'neck' \
# 	--batch_size 1 \
# 	--decoder_groups 32 \
# 	--len_snippet 32 \
# 	--save_path '/home/sid/' \
# 	--checkpoint_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/vinet--/vinet_rootgrouped_32_etmd_split3_bs4.pt' \
# 	--metrics_save_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/metrics_results'

# dataset="Hollywood2"
# cd ~/SaliencyModel/EEAA/SaliencyModel/
# python vinet_inferences.py --dataset $dataset \
# 	--videos_root_path /home/sid/DHF1K \
# 	--train_path_data /home/sid/$dataset/annotation \
# 	--val_path_data /home/sid/$dataset/val \
# 	--split 1 \
# 	--video_frame_class_file_path /home/sid/mvva_dataset/video_frame_num_rois_class_dict.pickle \
# 	--neck_name 'neck' \
# 	--batch_size 1 \
# 	--decoder_groups 32 \
# 	--len_snippet 32 \
# 	--save_path '/home/sid/' \
# 	--checkpoint_path '/home/sid/ViNet-Saliency/vinet_rootgrouped_32_bs8_kld_cc.pt' \
# 	--metrics_save_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/metrics_results'

# python vinet_inferences.py --dataset $dataset \
#     --videos_root_path /home/sid/UCF \
#     --train_path_data /home/sid/UCF/training \
# 	--val_path_data /home/sid/UCF/testing \
#     --neck 'neck' \
#     --batch_size 1 \
#     --decoder_groups 32 \
#     --len_snippet 32 \
#     --alternate 1 \
# 	--checkpoint_path '/home/sid/ViNet-Saliency/vinet_rootgrouped_32_ucf_fine-tune_bs8.pt' \
#     --metrics_save_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/metrics_results' \
#     --subset_type 'all'
    # --modelA_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/EEAA-B_random1_DHF1K_dissimilarity_neck_32_channel_shuffle_4_bs_dissimilarity_neck_1gpus_sid_human_best_0.5s_modelA_rerun1.pt \
    # --modelB_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/EEAA-B_random1_DHF1K_image_saliency_neck_8_channel_shuffle_4_bs_image_saliency_neck_1gpus_sid_non_human_best_0.5s_modelB.pt \
    # --subset_type 'all' \
    # --use_image_saliency 1



    #EEAA-B_random1_DHF1K_ensemble_neck_32_channel_shuffle_6_bs_ensemble_neck_1gpus_sid_ensemble_decoder_image_saliency_modelC


    #EEAA-B_random1_DHF1K_ensemble_neck_32_channel_shuffle_6_bs_ensemble_neck_1gpus_sid_ensemble_modelC

    # --modelA_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/EEAA-B_random1_DHF1K_neck_32_channel_shuffle_4_bs_neck_1gpus_sid_human_modelA.pt \
    # --modelB_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/EEAA-B_random1_DHF1K_image_saliency_neck_8_channel_shuffle_4_bs_neck_1gpus_sid_non_human_image_saliency_modelB_best.pt \


# The best model ensemble model till now is EEAA-B_random1_DHF1K_ensemble_neck_32_channel_shuffle_6_bs_ensemble_neck_1gpus_sid_ensemble_decoder_image_saliency_neck_rerun1_modelC.pt