#!/bin/bash

echo "activating environment"
cd
source activate DL



echo "starting inferences_metrics.py"
cd ~/SaliencyModel/EEAA/SaliencyModel/



# dataset="DIEM"
# video_name='sport_wimbledon_federer_final_1280x704'
# # video_file='700.AVI'

# python inferences_heatmap.py --dataset $dataset \
# --videos_root_path /home/sid/Audio-Visual-SaliencyDatasets  \
# --video_frames_root_path /home/sid/Audio-Visual-SaliencyDatasets/video_frames/$dataset/$video_name \
# --model 'gt' \
# --video_name $video_name \
# --sal_maps_root_path /home/sid/Audio-Visual-SaliencyDatasets/annotations/$dataset/ \
# --sal_maps_path /home/sid/Audio-Visual-SaliencyDatasets/annotations/$dataset/$video_name/maps \
# --video_frames_path /home/sid/Audio-Visual-SaliencyDatasets/video_frames/$dataset/$video_name \
# --model_tag 'gt' \



# python inferences_heatmap.py --dataset $dataset \
# --videos_root_path /home/sid/Audio-Visual-SaliencyDatasets \
# --video_frames_root_path /home/sid/Audio-Visual-SaliencyDatasets/video_frames/$dataset/$video_name \
# --model 'ViNet-S' \
# --video_name $video_name \
# --sal_maps_root_path /home/sid/inferences/ \
# --sal_maps_path /home/sid/inferences/'ViNet-S'/$video_name \
# --video_frames_path /home/sid/Audio-Visual-SaliencyDatasets/video_frames/$dataset/$video_name \
# --model_tag 'ViNet-S'



# python inferences_heatmap.py --dataset $dataset \
# --videos_root_path /home/sid/Audio-Visual-SaliencyDatasets \
# --video_frames_root_path /home/sid/Audio-Visual-SaliencyDatasets/video_frames/$dataset/$video_name \
# --model 'ViNet-A' \
# --video_name $video_name \
# --sal_maps_root_path /home/sid/inferences/ \
# --sal_maps_path /home/sid/inferences/'ViNet-A'/$video_name \
# --video_frames_path /home/sid/Audio-Visual-SaliencyDatasets/video_frames/$dataset/$video_name \
# --model_tag 'ViNet-A'




# python inferences_heatmap.py --dataset $dataset \
# --videos_root_path /home/sid/Audio-Visual-SaliencyDatasets \
# --video_frames_root_path /home/sid/Audio-Visual-SaliencyDatasets/video_frames/$dataset/$video_name \
# --model 'ViNet-E' \
# --video_name $video_name \
# --sal_maps_root_path /home/sid/inferences/ \
# --sal_maps_path /home/sid/inferences/'ViNet-E'/$video_name \
# --video_frames_path /home/sid/Audio-Visual-SaliencyDatasets/video_frames/$dataset/$video_name \
# --model_tag 'ViNet-E'




# python inferences_heatmap.py --dataset $dataset \
# --videos_root_path /home/sid/Audio-Visual-SaliencyDatasets \
# --video_frames_root_path /home/sid/Audio-Visual-SaliencyDatasets/video_frames/$dataset/$video_name \
# --model 'STSANet' \
# --video_name $video_name \
# --sal_maps_root_path /home/sid/inferences/ \
# --sal_maps_path /home/sid/inferences/'STSANet_fine-tuned_on_DIEM'/$video_name \
# --video_frames_path /home/sid/Audio-Visual-SaliencyDatasets/video_frames/$dataset/$video_name \
# --model_tag 'STSANet' \


dataset="UCF"
video_name='Kicking-Front-004'
# video_file='700.AVI'

python inferences_heatmap.py --dataset $dataset \
--videos_root_path /home/sid/UCF \
--video_frames_root_path /home/sid/UCF/testing \
--model 'gt' \
--video_name $video_name \
--sal_maps_root_path /home/sid/UCF/testing \
--sal_maps_path /home/sid/UCF/testing/$video_name/maps \
--video_frames_path /home/sid/UCF/testing/$video_name/images \
--model_tag 'gt' \



python inferences_heatmap.py --dataset $dataset \
--videos_root_path /home/sid/UCF \
--video_frames_root_path /home/sid/UCF/testing \
--model 'ViNet-S' \
--video_name $video_name \
--sal_maps_root_path /home/sid/inferences/ \
--sal_maps_path /home/sid/inferences/'ViNet-S'/$video_name \
--video_frames_path /home/sid/UCF/testing/$video_name/images \
--model_tag 'ViNet-S'



python inferences_heatmap.py --dataset $dataset \
--videos_root_path /home/sid/UCF \
--video_frames_root_path /home/sid/UCF/testing \
--model 'ViNet-A' \
--video_name $video_name \
--sal_maps_root_path /home/sid/inferences/ \
--sal_maps_path /home/sid/inferences/'ViNet-A'/$video_name \
--video_frames_path /home/sid/UCF/testing/$video_name/images \
--model_tag 'ViNet-A'




python inferences_heatmap.py --dataset $dataset \
--videos_root_path /home/sid/UCF \
--video_frames_root_path /home/sid/UCF/testing \
--model 'ViNet-E' \
--video_name $video_name \
--sal_maps_root_path /home/sid/inferences/ \
--sal_maps_path /home/sid/inferences/'ViNet-E'/$video_name \
--video_frames_path /home/sid/UCF/testing/$video_name/images \
--model_tag 'ViNet-E'




python inferences_heatmap.py --dataset $dataset \
--videos_root_path /home/sid/UCF \
--video_frames_root_path /home/sid/UCF/testing \
--model 'STSANet' \
--video_name $video_name \
--sal_maps_root_path /home/sid/inferences/ \
--sal_maps_path /home/sid/inferences/'STSANet_fine-tuned_on_UCF'/$video_name \
--video_frames_path /home/sid/UCF/testing/$video_name/images \
--model_tag 'STSANet' \



# dataset="DHF1K"
# video_name='0700'
# video_file='700.AVI'

# python inferences_heatmap.py --dataset $dataset \
# --videos_root_path /home/sid/DHF1K/video \
# --video_frames_root_path /home/sid/DHF1K/val \
# --model 'gt' \
# --video_file $video_file \
# --video_name $video_name \
# --sal_maps_root_path /home/sid/DHF1K/val \
# --sal_maps_path /home/sid/DHF1K/val/$video_name/maps \
# --video_frames_path /home/sid/DHF1K/val/$video_name/images \
# --model_tag 'gt' \



# python inferences_heatmap.py --dataset $dataset \
# --videos_root_path /home/sid/DHF1K/video \
# --video_frames_root_path /home/sid/DHF1K/val \
# --model 'ViNet-S' \
# --video_file $video_file \
# --video_name $video_name \
# --sal_maps_root_path /home/sid/inferences/ \
# --sal_maps_path /home/sid/inferences/'ViNet-S'/$video_name \
# --video_frames_path /home/sid/DHF1K/val/$video_name/images \
# --model_tag 'ViNet-S'



# python inferences_heatmap.py --dataset $dataset \
# --videos_root_path /home/sid/DHF1K/video \
# --video_frames_root_path /home/sid/DHF1K/val \
# --model 'ViNet-A' \
# --video_file $video_file \
# --video_name $video_name \
# --sal_maps_root_path /home/sid/inferences/ \
# --sal_maps_path /home/sid/inferences/'ViNet-A'/$video_name \
# --video_frames_path /home/sid/DHF1K/val/$video_name/images \
# --model_tag 'ViNet-A'




# python inferences_heatmap.py --dataset $dataset \
# --videos_root_path /home/sid/DHF1K/video \
# --video_frames_root_path /home/sid/DHF1K/val \
# --model 'ViNet-E' \
# --video_file $video_file \
# --video_name $video_name \
# --sal_maps_root_path /home/sid/inferences/ \
# --sal_maps_path /home/sid/inferences/'ViNet-E'/$video_name \
# --video_frames_path /home/sid/DHF1K/val/$video_name/images \
# --model_tag 'ViNet-E'




# python inferences_heatmap.py --dataset $dataset \
# --videos_root_path /home/sid/DHF1K/video \
# --video_frames_root_path /home/sid/DHF1K/val \
# --model 'STSANet' \
# --video_file $video_file \
# --video_name $video_name \
# --sal_maps_root_path /home/sid/inferences/ \
# --sal_maps_path /home/sid/inferences/'STSANet_DHF1K'/$video_name \
# --video_frames_path /home/sid/DHF1K/val/$video_name/images \
# --model_tag 'STSANet_DHF1K' \

















