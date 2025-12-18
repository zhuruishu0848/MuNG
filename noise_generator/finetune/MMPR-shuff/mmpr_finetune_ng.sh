#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_CUDA_ARCH_LIST="8.0"
DIR=`pwd`

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="./checkpoints/base/Qwen2.5-VL-3B-Instruct" 
DATA="datasets/MMPR-v1.1/shuff_mmpr.jsonl"
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS noise_generator/mmpr_finetune_ng_shuff.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --bf16 True \
    --fix_vit True \
    --output_dir checkpoints/MMPR-shuff-3B/noise-5e4-n8\
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --ng_heads 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --learning_rate 5e-4 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to wandb \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --train_noise_generator True \
    --deepspeed noise_generator/finetune/ds_config_zero3.json