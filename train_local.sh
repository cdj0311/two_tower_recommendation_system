#!/bin/sh
ckpt_dir=./ckpt
model_dir=./model

train_data=train_files.txt
eval_data=eval_files.txt
train_steps=100000000
batch_size=512
learning_rate=0.0005
save_steps=100000

python main.py \
    --train_data=${train_data} \
    --eval_data=${eval_data} \
    --model_dir=${ckpt_dir} \
    --output_model=${model_dir} \
    --train_steps=${train_steps} \
    --save_checkpoints_steps=${save_steps} \
    --learning_rate=${learning_rate} \
    --batch_size=${batch_size} \
    --train_on_cluster=False \
    --export_model=True \
    --gpuid=1