#!/bin/bash
#SBATCH --job-name=acar_train_mvva
#SBATCH -A research
#SBATCH -c 28
#SBATCH --gres=gpu:3
#SBATCH -o /home/girmaji08/EEAA/SaliencyModel/logs/inferences_metrics_run1.log
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END


echo "activating environment"
cd 
# source env_sal/bin/activate
source activate DL
# module load u18/cuda/11.6
# module load u18/cudnn/8.4.0-cuda-11.6
# module load u18/matlab/R2022a
a=$USER

# parent_directory="/ssd_scratch/cvit/girmaji08"
# echo "copying dataset"
# dataset_path="Dataset/video_frames/"

################################################################################################################ 


dataset="MVVA"



# mkdir -p "$parent_directory/$dataset_path/"
# if [ ! -d "$parent_directory/$dataset_path/mvva" ]; then

# 	mkdir -p $ssd_path

# 	mkdir -p $ssd_path/Dataset/annotations
# 	mkdir -p $ssd_path/Dataset/video_audio
# 	mkdir -p $ssd_path/Dataset/video_frames

# 	rsync -r girmaji08@ada.iiit.ac.in:/share3/girmaji08/mvva_dataset/annotations/mvva $ssd_path/Dataset/annotations

# 	rsync -r girmaji08@ada.iiit.ac.in:/share3/girmaji08/mvva_dataset/video_audio/mvva $ssd_path/Dataset/video_audio

# 	rsync -r girmaji08@ada.iiit.ac.in:/share3/girmaji08/mvva_dataset/video_frames/mvva $ssd_path/Dataset/video_frames

# 	rsync -r girmaji08@ada.iiit.ac.in:/share3/girmaji08/mvva_dataset/fold_lists $ssd_path/Dataset/

# 	rsync -r girmaji08@ada.iiit.ac.in:/share3/girmaji08/mvva_raw_videos $ssd_path/Dataset/

# 	rsync -r girmaji08@ada.iiit.ac.in:/share3/girmaji08/fixation_data_mvva $ssd_path

# fi

# rsync -r girmaji08@ada.iiit.ac.in:/share3/girmaji08/fixation_data_mvva $ssd_path

echo "starting inferences_metrics.py"
cd ~/SaliencyModel/EEAA/SaliencyModel/

python inferences_metrics.py --dataset mvva \
	--videos_frames_root_path /home/sid/mvva_dataset/video_frames/mvva \
	--videos_root_path /home/sid/mvva_dataset/mvva_raw_videos \
	--gt_sal_maps_path /home/sid/mvva_dataset/annotations/mvva \
	--fold_lists_path /home/sid/mvva_dataset/fold_lists \
	--fixation_data_path /home/sid/mvva_dataset/fixation_data_mvva \
	--audio_maps_path /ssd_scratch/cvit/girmaji08/audio_spatial_maps \
	--split 1 \
	--video_frame_class_file_path /home/sid/mvva_dataset/video_frame_num_rois_class_dict.pickle \
	--neck_name 'neck' \
	--batch_size 1 \
	--decoder_groups 32 \
	--save_path '/home/sid/' \
	--checkpoint_path '/home/sid/SaliencyModel/testSal/mvva_split1_BiFPN_-0.2926.pt' \
	--metrics_save_path '/home/sid/SaliencyModel/EEAA/SaliencyModel/metrics_results'