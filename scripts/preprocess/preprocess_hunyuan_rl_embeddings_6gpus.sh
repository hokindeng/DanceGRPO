#!/bin/bash

# Preprocessing script for HunyuanVideo embeddings - optimized for 6 GPUs

GPU_NUM=6  # Set to 6 GPUs
MODEL_PATH="data/HunyuanVideo"
OUTPUT_DIR="data/rl_embeddings"

echo "=========================================="
echo "HunyuanVideo Preprocessing - 6 GPU Config"
echo "=========================================="
echo "This will create text embeddings for training"
echo "Using 6 GPUs: GPU 0-5"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Copy tokenizer files to text encoder directories
echo "Copying tokenizer files..."
cp -rf data/HunyuanVideo/tokenizer/* data/HunyuanVideo/text_encoder
cp -rf data/HunyuanVideo/tokenizer_2/* data/HunyuanVideo/text_encoder_2

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting preprocessing with 6 GPUs..."
torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    fastvideo/data_preprocess/preprocess_hunyuan_embeddings.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "./assets/video_prompts.txt" \
    --model_type hunyuan_hf

echo "=========================================="
echo "Preprocessing complete!"
echo "Embeddings saved to: $OUTPUT_DIR"
echo "Files created:"
ls -lh $OUTPUT_DIR/
echo "=========================================="

