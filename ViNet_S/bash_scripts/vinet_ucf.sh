#!/bin/bash
#SBATCH -A research
#SBATCH --partition=long
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=bhav.beri@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --output=vinet_output_UCF.txt
#SBATCH -c 9
#SBATCH -n 1

echo "Activating environment"
source activate DL

cd ~/ViNet-Saliency/

echo "starting train.py"
# python3 train.py --train_path_data /home/sid/UCF/training --val_path_data /home/sid/UCF/testing --dataset UCF --frames_path images --use_wandb True --batch_size 4 --root_grouping True --grouped_conv True --load_weight /home/sid/testSal/vinet_rootgrouped_32.pt --model_val_path vinet_rootgrouped_32_ucf_fine-tune.pt --no_epochs 50
python3 train.py --train_path_data /home/sid/UCF/training --val_path_data /home/sid/UCF/testing --dataset UCF --frames_path images --batch_size 8 --root_grouping True --grouped_conv True --load_weight /home/sid/SaliencyModel/testSal/vinet_rootgrouped_32.pt --model_val_path vinet_rootgrouped_32_ucf_fine-tune_bs8.pt --no_epochs 100

echo "Done"