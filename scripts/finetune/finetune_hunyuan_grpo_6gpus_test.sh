#!/bin/bash

# Quick test script for HunyuanVideo GRPO on 6x H100 GPUs
# Use this to verify everything works before full training

export WANDB_DISABLED=true
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online

# Install dependencies
pip3 install moviepy
mkdir -p videos
pip3 install huggingface_hub==0.24.0 
pip3 install tf-keras==2.19.0
pip3 install trl==0.16.0
pip3 install transformers==4.46.1
pip3 install protobuf==5.29.5

echo "=========================================="
echo "HunyuanVideo GRPO TEST RUN - 6 GPU Config"
echo "=========================================="
echo "This is a quick test with minimal settings"
echo "Resolution: 256x256, 33 frames"
echo "Only 10 training steps"
echo "Expected time: ~15 minutes"
echo "=========================================="

# Single node, 6 GPUs, minimal settings for testing
torchrun --nnodes=1 --nproc_per_node=6 \
    fastvideo/train_grpo_hunyuan.py \
    --seed 42 \
    --model_type "hunyuan_hf" \
    --pretrained_model_name_or_path data/HunyuanVideo \
    --vae_model_path data/HunyuanVideo \
    --cache_dir data/.cache \
    --data_json_path data/rl_embeddings/videos2caption.json \
    --gradient_checkpointing \
    --selective_checkpointing 0.3 \
    --use_cpu_offload \
    --train_batch_size 1 \
    --sp_size 2 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 2 \
    --gradient_accumulation_steps 4 \
    --max_train_steps 10 \
    --learning_rate 8e-6 \
    --mixed_precision bf16 \
    --checkpointing_steps 5 \
    --validation_steps 100000000 \
    --checkpoints_total_limit 1 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir data/outputs/grpo_hunyuan_test \
    --tracker_project_name grpo_hunyuan_test \
    --h 256 \
    --w 256 \
    --t 33 \
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
    --timestep_fraction 0.5 \
    --use_same_noise \
    --bestofn 4 \
    --vq_coef 1.0 \
    --mq_coef 0.0

echo "=========================================="
echo "Test Run Complete!"
echo "If successful, run the full training script:"
echo "bash scripts/finetune/finetune_hunyuan_grpo_6gpus.sh"
echo "=========================================="

