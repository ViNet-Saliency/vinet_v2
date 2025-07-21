
source activate DL


dataset="Coutrot_db2"



echo "copying dataset"


echo "starting train.py"
cd /home/sid/SaliencyModel/EEAA/SaliencyModel



python train.py --dataset $dataset \
    --videos_root_path /home/sid/Audio-Visual-SaliencyDatasets \
    --cc_coeff -1 \
    --neck_name 'neck' \
    --split 1 \
    --batch_size 3 \
    --len_snippet 64 \
    --decoder_groups 32 \
    --model_tag 'neck_1gpus_sid_no_extra_data_0.5s_baseline_dhf1k_init_64clipsize_rndseed_best_run1' \
    --no_epochs 150 \
    --subset_type 'all' \
    --use_image_saliency 0 \
    --reload_data_every_epoch 0 \
    --use_action_classification 0 \
    --checkpoint_path /home/sid/SaliencyModel/testSal/DHF1K_baseline_32g_0.85294.pt
    # --checkpoint_path /home/sid/SaliencyModel/EEAA/SaliencyModel/saved_models/EEAA-B_1_mvva_neck_32_channel_shuffle4bs.pt

    #