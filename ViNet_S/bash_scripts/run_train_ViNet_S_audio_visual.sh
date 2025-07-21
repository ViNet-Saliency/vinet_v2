#!/bin/bash

source activate DL

echo "starting train.py"



dataset="ETMD_av"
dataset_root_path="/mnt/Shared-Storage/rohit"
root_folder_path="/mnt/Shared-Storage/rohit/Home"
fold_lists_root_path="$root_folder_path/ICASSP_Saliency"

cd $root_folder_path/ICASSP_Saliency/ViNet_S
echo "starting train.py"

CUDA_VISIBLE_DEVICES=1, python3 train.py --videos_root_path $dataset_root_path/Audio-Visual-SaliencyDatasets \
--dataset $dataset \
--batch_size 2 \
--root_grouping True \
--grouped_conv True \
--split 1 \
--fold_lists_path $fold_lists_root_path/fold_lists \
--checkpoint_path $root_folder_path/ICASSP_Saliency/ViNet_S/saved_models/DHF1K_vinet_s_rootgrouped_32_bs8_kld_cc.pt \
--model_save_path $root_folder_path/ICASSP_Saliency/ViNet_S/saved_models/${dataset}_vinet_s_rootgrouped_32_bs8_kld_cc.pt \
--no_epochs 120

echo "Done"








# #!/bin/bash

# echo "activating environment"
# source activate DL

# cd ~/ViNet-Saliency/

# echo "starting train.py"
# python3 train.py --dataset AVAD \
# --split 1 \
# --batch_size 2 \
# --root_grouping True \
# --grouped_conv True \
# --load_weight /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/vinet--/vinet_rootgrouped_32_dhf1k.pt \
# --model_val_path vinet_rootgrouped_32_avad_split1_seed3.pt \
# --no_epochs 120

# echo "Done"