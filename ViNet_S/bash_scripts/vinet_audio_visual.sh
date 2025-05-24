#!/bin/bash

echo "activating environment"
source activate DL

cd ~/ViNet-Saliency/

echo "starting train.py"
python3 train.py --dataset AVAD --split 1 --batch_size 2 --root_grouping True --grouped_conv True --load_weight /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/vinet--/vinet_rootgrouped_32_dhf1k.pt --model_val_path vinet_rootgrouped_32_avad_split1_seed3.pt --no_epochs 120

echo "Done"