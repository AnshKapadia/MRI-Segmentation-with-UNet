project_path='C:\\Users\\anshk\\MRI_segmentation'

python main.py \
    --project_dir ${project_path} \
    --train_path ${project_path}/data/dummy_train \
    --eval_path ${project_path}/data/dummy_eval \
    --save_path ${project_path}/checkpoints \
    --max_epoch 150 \
    --inference_interval 5 \
    --lr 0.0001 \
    --test_step 4 \
    --lr_decay 0.95 \
    --train True
