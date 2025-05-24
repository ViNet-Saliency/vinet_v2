#!/bin/bash
#SBATCH -A research
#SBATCH --partition=long
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --mail-user=bhav.beri@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --output=vinet_output_combined.txt
#SBATCH -c 15
#SBATCH -n 1

echo "Activating environment"
source activate DL

cd ~/ViNet-Saliency/

echo "starting train.py"
# python3 train.py --train_path_data /ssd_scratch/cvit/sarthak395/DHF1K/annotation --val_path_data /ssd_scratch/cvit/sarthak395/DHF1K/val --dataset DHF1KDataset --batch_size 4 --use_wandb True --root_grouping True --grouped_conv True --model_val_path vinet_rootgrouped_32.pt
python3 train.py --train_path_data ~/DHF1K/annotation --val_path_data ~/DHF1K/val --frames_path images --dataset DHF1KDataset --batch_size 8 --root_grouping True --grouped_conv True --model_val_path vinet_rootgrouped_32_bs8_kld_cc.pt --no_epochs 120

echo "Done"