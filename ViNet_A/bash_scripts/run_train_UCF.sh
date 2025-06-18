
source activate DL

dataset="UCF"
dataset_root_path="/mnt/Shared-Storage/rohit"
root_folder_path="/mnt/Shared-Storage/rohit/Home"
fold_lists_root_path="$root_folder_path/ICASSP_Saliency"

cd $root_folder_path/ICASSP_Saliency/ViNet_A

CUDA_VISIBLE_DEVICES=1, python train.py --dataset $dataset \
    --videos_root_path $dataset_root_path/UCF \
    --cc_coeff -1 \
    --neck_name 'neck' \
    --batch_size 6 \
    --len_snippet 64 \
    --decoder_groups 32 \
    --no_epochs 150 \
    --use_action_classification 0 \
    --model_save_path $root_folder_path/ICASSP_Saliency/ViNet_A/saved_models \
    --checkpoint_path $root_folder_path/ICASSP_Saliency/ViNet_A/saved_models/ViNet_A_DHF1K_no_split_neck_32_kldiv_cc___6_0.pt