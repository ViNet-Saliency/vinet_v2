#!/bin/bash

source activate DL

echo "starting train.py"



dataset="UCF"
dataset_root_path="/mnt/Shared-Storage/rohit"
root_folder_path="/mnt/Shared-Storage/rohit/Home"
fold_lists_root_path="$root_folder_path/ICASSP_Saliency"

cd $root_folder_path/ICASSP_Saliency/ViNet_S
echo "starting train.py"

CUDA_VISIBLE_DEVICES=1, python3 train.py --videos_root_path $dataset_root_path/UCF \
--frames_path images \
--dataset $dataset \
--batch_size 8 \
--root_grouping True \
--grouped_conv True \
--checkpoint_path $root_folder_path/ICASSP_Saliency/ViNet_S/saved_models/DHF1K_vinet_s_rootgrouped_32_bs8_kld_cc.pt \
--model_save_path $root_folder_path/ICASSP_Saliency/ViNet_S/saved_models/${dataset}_vinet_s_rootgrouped_32_bs8_kld_cc.pt \
--no_epochs 120

echo "Done"










# #!/bin/bash
# #SBATCH -A research
# #SBATCH --partition=long
# #SBATCH --mem-per-cpu=2G
# #SBATCH --gres=gpu:1
# #SBATCH --time=20:00:00
# #SBATCH --mail-user=bhav.beri@research.iiit.ac.in
# #SBATCH --mail-type=ALL
# #SBATCH --output=vinet_output_Hollywood.txt
# #SBATCH -c 9
# #SBATCH -n 1

# echo "Activating environment"
# source activate DL

# cd ~/ViNet-Saliency/

# echo "starting train.py"
# # python3 train.py --train_path_data /ssd_scratch/cvit/sarthak395/Hollywood/training --val_path_data /ssd_scratch/cvit/sarthak395/Hollywood/testing --dataset Hollywood --frames_path images --batch_size 4 --use_wandb True --root_grouping True --grouped_conv True --load_model_path /ssd_scratch/cvit/sarthak395/DHF1K/saved_models/vinet_rootgrouped_32.pt --model_val_path vinet_rootgrouped_32_Hollywood.pt --no_epochs 20
# python3 train.py --train_path_data ~/Hollywood2/training \
# --val_path_data ~/Hollywood2/testing \
# --dataset Hollywood \
# --frames_path images \
# --batch_size 8 \
# --root_grouping True \
# --grouped_conv True \
# --load_weight ~/SaliencyModel/testSal/vinet_rootgrouped_32.pt \
# --model_val_path vinet_rootgrouped_32_Hollywood.pt \
# --no_epochs 120

# echo "Done"