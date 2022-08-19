python train.py \
    --model_dir /Share/home/qiyifan/filebase/source/mt5-base \
    --data_path ../data \
    --output_dir ../tmp/mt5-base-finetuned_whole_set \
    --max_length 512 \
    --batch_size 8 \
    --epoch 100 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --use_cuda \
    --save_checkpoint_all

python test.py \
    --model_dir ../tmp/mt5-base-finetuned_whole_set/checkpoint_99 \
    --data_path ../data \
    --output_dir ../tmp/mt5-base-finetuned_whole_set/checkpoint_99 \
    --max_length 512 \
    --batch_size 8 \
    --use_cuda