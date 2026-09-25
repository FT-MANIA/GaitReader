#!/bin/bash

# Usage: bash ./scripts/pretrain.sh <codebook_size> <dim_model>
# IMPORTANT: Make sure to change the data_root to your own path.
# For debugging, add the flag "--dev" to the end of the command.

codebook_size=$1
dim_model=$2
data_root=~/data/VQShape


python ./vqshape/pretrain.py \
    --data_root $data_root \
    --dim_embedding $dim_model \
    --normalize_length 512 \
    --patch_size 8 \
    --num_patch 64 \
    --num_token 64 \
    --num_transformer_enc_heads 8 \
    --num_transformer_enc_layers 4 \
    --num_tokenizer_heads 8 \
    --num_tokenizer_layers 4 \
    --num_transformer_dec_heads 8 \
    --num_transformer_dec_layers 2 \
    --num_code $codebook_size \
    --dim_code 8 \
    --codebook_type standard \
    --len_s 128 \
    --s_smooth_factor 1 \
    --lambda_x 1 \
    --lambda_z 1 \
    --lambda_s 1 \
    --lambda_dist 0.8 \
    --lambda_vq_commit 0.25 \
    --lambda_vq_entropy 0.1 \
    --entropy_gamma 1 \
    --lr 1e-4 \
    --batch_size 512 \
    --accumulate_grad_batches 4 \
    --gradient_clip 1 \
    --weight_decay 0.01 \
    --mask_ratio 0.25 \
    --warmup_step 1000 \
    --train_epoch 2 \
    --val_frequency 0.2 \
    --name uea_dim"$dim_model"_codebook"$codebook_size" \
    --num_nodes 1 \
    --num_devices 1 \
    --strategy "auto" \
    --precision "bf16-mixed" \
    --num_workers 8 \
    --balance_datasets
    