#!/bin/bash
#SBATCH -A research
#SBATCH --partition=long
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mail-user=bhav.beri@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --output=vinet_groups_output.txt
#SBATCH -c 9
#SBATCH -n 1

echo "Activated"

cd ~/ViNet-Saliency/

python3 train.py --train_path_data /ssd_scratch/cvit/sarthak395/DHF1K/annotation --val_path_data /ssd_scratch/cvit/sarthak395/DHF1K/val --dataset DHF1KDataset --batch_size 4 --use_wandb True --root_grouping True

echo "Done"
