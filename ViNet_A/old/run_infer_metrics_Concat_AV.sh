#!/bin/bash

echo "activating environment"
cd
source activate DL

dataset="Coutrot_db1"

echo "starting inferences_metrics_concat_AV.py"
cd ~/SaliencyModel/EEAA/SaliencyModel/





python inferences_metrics_concat_AV.py --dataset $dataset \
    --videos_root_path /home/sid/Audio-Visual-SaliencyDatasets \
	--videos_frames_root_path /home/sid/Audio-Visual-SaliencyDatasets/video_frames/$dataset \
	--gt_sal_maps_path /home/sid/Audio-Visual-SaliencyDatasets/annotations/$dataset \
	--fold_lists_path /home/sid/Audio-Visual-SaliencyDatasets/fold_lists \
	--fixation_data_path /home/sid/mvva_dataset/fixation_data_mvva \
	--audio_maps_path /ssd_scratch/cvit/girmaji08/audio_spatial_maps \
	--split 1 \
	--video_frame_class_file_path /home/sid/mvva_dataset/video_frame_num_rois_class_dict.pickle \
	--neck_name 'neck' \
	--batch_size 1 \
	--decoder_groups 32 \
	--len_snippet 64 \
	--save_path '/home/sid/' \
	--checkpoint_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/baseline/EEAA-B_1_Concat_AV_Dataset_neck_32_channel_shuffle_10_bs_concat_av_action_detection_64csz_32g_0.5s_removed_nans_noclip_10bs_run1.pt' \
	--metrics_save_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/metrics_results'