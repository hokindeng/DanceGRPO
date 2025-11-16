#!/bin/bash

# HunyuanVideo GRPO training WITHOUT reward model (for testing/debugging)
# Uses dummy rewards instead of VideoAlign

export WANDB_DISABLED=true
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "HunyuanVideo GRPO Training - NO REWARD MODEL"
echo "=========================================="
echo "WARNING: Training without reward model!"
echo "This is for testing memory/setup only"
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
    --selective_checkpointing 0.2 \
    --use_cpu_offload \
    --train_batch_size 1 \
    --sp_size 2 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 2 \
    --gradient_accumulation_steps 12 \
    --max_train_steps 202 \
    --learning_rate 8e-6 \
    --mixed_precision bf16 \
    --checkpointing_steps 50 \
    --validation_steps 100000000 \
    --checkpoints_total_limit 2 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir data/outputs/grpo_hunyuan_6gpu_no_reward \
    --tracker_project_name grpo_hunyuan_6gpu_no_reward \
    --h 384 \
    --w 384 \
    --t 49 \
    --sampling_steps 10 \
    --eta 0.25 \
    --fps 8 \
    --sampler_seed 1237 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --num_generations 8 \
    --shift 5 \
    --use_group \
    --timestep_fraction 0.5 \
    --use_same_noise \
    --bestofn 4 \
    --vq_coef 1.0 \
    --mq_coef 0.0

echo "=========================================="
echo "Training Complete!"
echo "=========================================="

