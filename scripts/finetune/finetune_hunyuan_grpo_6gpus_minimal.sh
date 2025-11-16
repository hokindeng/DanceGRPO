#!/bin/bash

# MINIMAL memory HunyuanVideo GRPO training for 6x H100 80GB GPUs
# Ultra-conservative settings to avoid OOM

export WANDB_DISABLED=true
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install dependencies
pip3 install moviepy
mkdir -p videos
pip3 install huggingface_hub==0.24.0 
pip3 install tf-keras==2.19.0
pip3 install trl==0.16.0
pip3 install transformers==4.46.1
pip3 install protobuf==5.29.5

echo "=========================================="
echo "HunyuanVideo GRPO Training - MINIMAL 6 GPU"
echo "=========================================="
echo "Hardware: 6x H100 80GB HBM3"
echo "Video Resolution: 256x256, 41 frames"
echo "Generations per prompt: 6"
echo "Expected memory per GPU: ~35-45 GB"
echo "=========================================="

# Single node, 6 GPUs
torchrun --nnodes=1 --nproc_per_node=6 \
    fastvideo/train_grpo_hunyuan.py \
    --seed 42 \
    --model_type "hunyuan_hf" \
    --pretrained_model_name_or_path data/HunyuanVideo \
    --vae_model_path data/HunyuanVideo \
    --cache_dir data/.cache \
    --data_json_path data/rl_embeddings/videos2caption.json \
    --gradient_checkpointing \
    --selective_checkpointing 0.1 \
    --use_cpu_offload \
    --train_batch_size 1 \
    --sp_size 2 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 1 \
    --gradient_accumulation_steps 16 \
    --max_train_steps 202 \
    --learning_rate 8e-6 \
    --mixed_precision bf16 \
    --checkpointing_steps 50 \
    --validation_steps 100000000 \
    --checkpoints_total_limit 2 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir data/outputs/grpo_hunyuan_6gpu_minimal \
    --tracker_project_name grpo_hunyuan_6gpu_minimal \
    --h 256 \
    --w 256 \
    --t 41 \
    --sampling_steps 8 \
    --eta 0.25 \
    --fps 8 \
    --sampler_seed 1237 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --num_generations 6 \
    --shift 5 \
    --use_group \
    --use_videoalign \
    --timestep_fraction 0.4 \
    --use_same_noise \
    --bestofn 3 \
    --vq_coef 1.0 \
    --mq_coef 0.0

echo "=========================================="
echo "Training Complete!"
echo "Checkpoints saved to: data/outputs/grpo_hunyuan_6gpu_minimal"
echo "=========================================="

