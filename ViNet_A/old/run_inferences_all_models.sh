#!/bin/bash

echo "activating environment"
cd
source activate DL

dataset="DIEM"

echo "starting inferences_metrics.py"
cd ~/SaliencyModel/EEAA/SaliencyModel/

video_name='sport_wimbledon_federer_final_1280x704'

python inferences_metrics.py --dataset $dataset \
    --videos_root_path /home/sid/Audio-Visual-SaliencyDatasets \
    --neck 'neck' \
    --batch_size 1 \
    --decoder_groups 32 \
    --len_snippet 64 \
    --alternate 1 \
    --metrics_save_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/metrics_results' \
    --subset_type 'all' \
    --save_path /home/sid \
    --save_inferences 1 \
    --video_names_list $video_name \
    --checkpoint_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/baseline/EEAA-B_DIEM_STAL_seed6161.pt' \


python vinet_inferences.py --dataset $dataset \
	--videos_root_path /home/sid/UCF \
	--train_path_data /home/sid/$dataset/training \
	--val_path_data /home/sid/$dataset/testing \
	--split 1 \
	--video_frame_class_file_path /home/sid/mvva_dataset/video_frame_num_rois_class_dict.pickle \
	--neck_name 'neck' \
	--batch_size 1 \
	--decoder_groups 32 \
	--len_snippet 32 \
	--save_path '/home/sid/' \
	--checkpoint_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/vinet--/vinet_rootgrouped_32_diem_bs4_kldiv_cc.pt' \
	--metrics_save_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/metrics_results' \
	--save_inferences 1 \
	--video_names_list $video_name


python vinet_prediction_avg.py --dataset $dataset \
	--videos_root_path /home/sid/Audio-Visual-SaliencyDatasets \
	--train_path_data /home/sid/$dataset/training \
	--val_path_data /home/sid/$dataset/testing \
	--split 1 \
	--video_frame_class_file_path /home/sid/mvva_dataset/video_frame_num_rois_class_dict.pickle \
	--neck_name 'neck' \
	--batch_size 1 \
	--decoder_groups 32 \
	--len_snippet 32 \
	--save_path '/home/sid/' \
	--checkpoint_path_1 '/home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/baseline/EEAA-B_DIEM_STAL_seed6161.pt' \
	--checkpoint_path_2 '/home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/vinet--/vinet_rootgrouped_32_diem_bs4_kldiv_cc.pt' \
	--metrics_save_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/metrics_results' \
	--subset_type 'all' \
	--save_inferences 1 \
	--video_names_list $video_name


python stsanet_inferences.py --dataset $dataset \
    --videos_root_path /home/sid/DHF1K \
    --neck 'neck' \
    --batch_size 1 \
    --decoder_groups 32 \
    --len_snippet 64 \
    --alternate 1 \
	--checkpoint_path '/home/sid/STSANet/STSANet_fine-tuned_on_DIEM.pth' \
    --metrics_save_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/metrics_results' \
    --subset_type 'all' \
    --save_path /home/sid/ \
    --save_inferences 1 \
    --video_names_list $video_name










# video_name='Kicking-Front-004'

# python inferences_metrics.py --dataset $dataset \
#     --videos_root_path /home/sid/UCF \
#     --neck 'neck' \
#     --batch_size 1 \
#     --decoder_groups 32 \
#     --len_snippet 64 \
#     --alternate 1 \
#     --metrics_save_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/metrics_results' \
#     --subset_type 'all' \
#     --save_path /home/sid \
#     --save_inferences 1 \
#     --video_names_list $video_name \
#     --checkpoint_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/baseline/EEAA-B_UCF_action_detection_64csz.pt' \


# python vinet_inferences.py --dataset $dataset \
# 	--videos_root_path /home/sid/UCF \
# 	--train_path_data /home/sid/$dataset/training \
# 	--val_path_data /home/sid/$dataset/testing \
# 	--split 1 \
# 	--video_frame_class_file_path /home/sid/mvva_dataset/video_frame_num_rois_class_dict.pickle \
# 	--neck_name 'neck' \
# 	--batch_size 1 \
# 	--decoder_groups 32 \
# 	--len_snippet 32 \
# 	--save_path '/home/sid/' \
# 	--checkpoint_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/vinet--/vinet_rootgrouped_32_ucf_fine-tune_bs8.pt' \
# 	--metrics_save_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/metrics_results' \
# 	--save_inferences 1 \
# 	--video_names_list $video_name


# python vinet_prediction_avg.py --dataset $dataset \
# 	--videos_root_path /home/sid/UCF \
# 	--train_path_data /home/sid/$dataset/training \
# 	--val_path_data /home/sid/$dataset/testing \
# 	--split 1 \
# 	--video_frame_class_file_path /home/sid/mvva_dataset/video_frame_num_rois_class_dict.pickle \
# 	--neck_name 'neck' \
# 	--batch_size 1 \
# 	--decoder_groups 32 \
# 	--len_snippet 64 \
# 	--save_path '/home/sid/' \
# 	--checkpoint_path_1 '/home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/baseline/EEAA-B_UCF_action_detection_64csz.pt' \
# 	--checkpoint_path_2 '/home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/vinet--/vinet_rootgrouped_32_ucf_fine-tune_bs8.pt' \
# 	--metrics_save_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/metrics_results' \
# 	--subset_type 'all' \
# 	--save_inferences 1 \
# 	--video_names_list $video_name

# python stsanet_inferences.py --dataset $dataset \
#     --videos_root_path /home/sid/UCF \
#     --neck 'neck' \
#     --batch_size 1 \
#     --decoder_groups 32 \
#     --len_snippet 64 \
#     --alternate 1 \
# 	--checkpoint_path '/home/sid/STSANet/STSANet_fine-tuned_on_UCF.pth' \
#     --metrics_save_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/metrics_results' \
#     --subset_type 'all' \
#     --save_path /home/sid/ \
#     --save_inferences 1 \
#     --video_names_list $video_name



# dataset="DHF1K"

# echo "starting inferences_metrics.py"
# cd ~/SaliencyModel/EEAA/SaliencyModel/


# video_name='0700'

# python inferences_metrics.py --dataset $dataset \
#     --videos_root_path /home/sid/DHF1K \
#     --neck 'neck' \
#     --batch_size 1 \
#     --decoder_groups 32 \
#     --len_snippet 64 \
#     --alternate 1 \
#     --metrics_save_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/metrics_results' \
#     --subset_type 'all' \
#     --save_path /home/sid \
#     --save_inferences 1 \
#     --video_names_list $video_name \
#     --checkpoint_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/baseline/DHF1K_STAL_32g_6bs_0.85294.pt' \


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
# 	--checkpoint_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/vinet--/vinet_rootgrouped_32_dhf1k.pt' \
# 	--metrics_save_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/metrics_results' \
# 	--save_inferences 1 \
# 	--video_names_list $video_name


# python vinet_prediction_avg.py --dataset $dataset \
# 	--videos_root_path /home/sid/DHF1K \
# 	--train_path_data /home/sid/$dataset/annotation \
# 	--val_path_data /home/sid/$dataset/val \
# 	--split 1 \
# 	--video_frame_class_file_path /home/sid/mvva_dataset/video_frame_num_rois_class_dict.pickle \
# 	--neck_name 'neck' \
# 	--batch_size 1 \
# 	--decoder_groups 32 \
# 	--len_snippet 64 \
# 	--save_path '/home/sid/' \
# 	--checkpoint_path_1 '/home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/baseline/DHF1K_STAL_32g_6bs_0.85294.pt' \
# 	--checkpoint_path_2 '/home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/vinet--/vinet_rootgrouped_32_dhf1k.pt' \
# 	--metrics_save_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/metrics_results' \
# 	--subset_type 'all' \
# 	--save_inferences 1 \
# 	--video_names_list $video_name

# python stsanet_inferences.py --dataset $dataset \
#     --videos_root_path /home/sid/DHF1K \
#     --neck 'neck' \
#     --batch_size 1 \
#     --decoder_groups 32 \
#     --len_snippet 64 \
#     --alternate 1 \
# 	--checkpoint_path '/home/sid/STSANet/STSANet_DHF1K.pth' \
#     --metrics_save_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/metrics_results' \
#     --subset_type 'all' \
#     --save_path /home/sid/ \
#     --save_inferences 1 \
#     --video_names_list $video_name