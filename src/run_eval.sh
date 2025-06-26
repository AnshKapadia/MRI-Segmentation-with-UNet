project_path='C:\\Users\\anshk\\MRI_segmentation'

python main.py \
    --project_dir ${project_path} \
    --train_path ${project_path}/data/training_data1_v2 \
    --eval_path ${project_path}/data/validation_data \
    --save_path ${project_path}/checkpoints \
    --max_epoch 150 \
    --inference_interval 50 \
    --lr 0.0001 \
    --test_step 4 \
    --lr_decay 0.95 \
    --eval True \
    --init_model ${project_path}/checkpoints/unet_brats_94.pt
