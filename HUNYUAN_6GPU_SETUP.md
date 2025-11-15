# HunyuanVideo Training on 6× H100 GPUs

## 🎯 Overview

This guide shows how to train **HunyuanVideo (Text-to-Video)** using GRPO on your 6× NVIDIA H100 80GB HBM3 GPUs.

**What is HunyuanVideo?**
- **Model Type**: Text-to-Video (T2V) generation
- **Input**: Text prompt (e.g., "A cat walking in the snow")
- **Output**: Generated video matching the description
- **Training Method**: GRPO (Group Relative Policy Optimization) with VideoAlign reward model

## 📊 Configuration Comparison

| Setting | Original (32 GPUs) | Your Setup (6 GPUs) | Reason |
|---------|-------------------|---------------------|---------|
| **GPUs** | 32 H800 | 6 H100 | Your hardware |
| **Resolution** | 480×480 | 384×384 | Reduce memory (-40%) |
| **Frames** | 53 | 49 | Reduce memory (-8%) |
| **Sampling Steps** | 16 | 12 | Reduce compute (-25%) |
| **Generations/prompt** | 24 | 12 | Reduce memory (-50%) |
| **Best-of-N** | 8 | 6 | Proportional to generations |
| **Gradient Accumulation** | 4 | 8 | Maintain effective batch size |
| **Sequence Parallel** | 1 | 2 | Split temporal dimension |
| **CPU Offload** | No | Yes | Save GPU memory |
| **Selective Checkpointing** | 1.0 | 0.3 | More aggressive (save 70%) |
| **Timestep Fraction** | 0.6 | 0.5 | Train on 50% of steps |

## ⚡ Performance Expectations

### Memory Usage (per GPU)
```
Model Parameters (FSDP shard):        ~12 GB
Activations (384×384×49):             ~15 GB
Gradients:                            ~12 GB
Optimizer States (CPU offloaded):     ~2 GB (GPU) + ~10 GB (CPU)
Video Decoding Buffer:                ~8 GB
───────────────────────────────────────────────
Total per GPU:                        ~49 GB / 80 GB ✅
Peak usage:                           ~55 GB / 80 GB ✅
```

### Training Speed
- **Time per step**: ~50-60 seconds (vs. ~12s with 32 GPUs)
- **Total steps**: 202
- **Total training time**: **~2.8-3.3 hours**
- **Checkpoint size**: ~20 GB each (saved every 50 steps)

### Expected Results
- ✅ **Algorithm**: Same GRPO, same reward model (VideoAlign)
- ✅ **Learning**: Will converge, just slower than 32 GPUs
- ⚠️ **Resolution**: 384×384 vs. 480×480 (slightly lower quality)
- ⚠️ **Convergence**: May need 10-20% more iterations
- ⚠️ **Exploration**: Fewer generations = less diversity per prompt

## 🚀 Quick Start

### Step 1: Download Model Checkpoints

```bash
# Create data directory
mkdir -p data

# Download HunyuanVideo (20GB, takes ~30 min)
cd data
git lfs install
git clone https://huggingface.co/hunyuanvideo-community/HunyuanVideo
cd ..

# Download VideoAlign reward model
mkdir -p videoalign_ckpt
# Follow instructions at: https://huggingface.co/KwaiVGI/VideoReward
```

### Step 2: Prepare Dataset

```bash
# Example: Create a simple dataset for testing
mkdir -p data/rl_embeddings/prompt_embed
mkdir -p data/rl_embeddings/prompt_attention_mask

# The preprocessing script will create embeddings from your video prompts
# See: assets/video_prompts.txt for example prompts
bash scripts/preprocess/preprocess_hunyuan_rl_embeddings.sh
```

### Step 3: Run Test Training (Recommended)

Start with a quick test to verify everything works:

```bash
# Test run: 10 steps, 256×256 resolution, ~15 minutes
bash scripts/finetune/finetune_hunyuan_grpo_6gpus_test.sh
```

**Monitor the test:**
```bash
# In another terminal, watch GPU usage
watch -n 1 nvidia-smi

# Check training logs
tail -f data/outputs/grpo_hunyuan_test/train.log
```

### Step 4: Run Full Training

If the test succeeds:

```bash
# Full training: 202 steps, 384×384 resolution, ~3 hours
bash scripts/finetune/finetune_hunyuan_grpo_6gpus.sh
```

## 📈 Monitoring Training

### Check Reward Progress

```bash
# Video Quality rewards (should increase over time)
tail -f vq_reward.txt

# Motion Quality rewards
tail -f mq_reward.txt
```

Expected VQ reward progression:
```
Initial: ~3.5-3.7
After 50 steps: ~3.8-4.0
After 100 steps: ~4.1-4.3
After 200 steps: ~4.3-4.5 (60% improvement)
```

### Monitor GPU Memory

```bash
# Watch memory usage
nvidia-smi dmon -s mu -d 1

# Or use gpustat (if installed)
gpustat -i 1
```

Expected memory per GPU:
- **Idle**: 12 GB (model loaded)
- **During rollout**: 45-50 GB (generating videos)
- **During training**: 50-55 GB (backprop + gradients)

### Check Generated Videos

Videos are saved during training:
```bash
ls -lh videos/
# You'll see: hunyuan_0_0.mp4, hunyuan_0_1.mp4, etc.
# One per GPU per prompt (6 GPUs = 6 videos per batch)
```

## 🔧 Troubleshooting

### Issue: CUDA Out of Memory (OOM)

**Solution 1**: Reduce resolution further
```bash
# Edit the script, change:
--h 384 --w 384 --t 49
# to:
--h 320 --w 320 --t 45
```

**Solution 2**: Reduce generations
```bash
# Change:
--num_generations 12 --bestofn 6
# to:
--num_generations 8 --bestofn 4
```

**Solution 3**: More aggressive checkpointing
```bash
# Change:
--selective_checkpointing 0.3
# to:
--selective_checkpointing 0.1  # Checkpoint only 10% of layers
```

### Issue: Training is too slow

**Check 1**: Verify GPU utilization
```bash
nvidia-smi dmon -s u -d 1
# Should see 80-95% GPU utilization during training
```

**Check 2**: Reduce sampling steps
```bash
# Change:
--sampling_steps 12
# to:
--sampling_steps 8  # Faster but slightly lower quality
```

**Check 3**: Reduce timestep fraction
```bash
# Change:
--timestep_fraction 0.5
# to:
--timestep_fraction 0.3  # Train on fewer timesteps
```

### Issue: Reward not improving

**Check 1**: Verify reward model is working
```bash
# Check logs for VideoAlign output
grep "vq_reward\|mq_reward" data/outputs/*/train.log
```

**Check 2**: Increase learning rate
```bash
# Change:
--learning_rate 8e-6
# to:
--learning_rate 1e-5
```

**Check 3**: Use more generations for exploration
```bash
# Change:
--num_generations 12
# to:
--num_generations 16
```

### Issue: Process crashes or hangs

**Solution 1**: Check NCCL communication
```bash
# Add these exports before training:
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

**Solution 2**: Reduce dataloader workers
```bash
# Change:
--dataloader_num_workers 2
# to:
--dataloader_num_workers 1
```

**Solution 3**: Increase timeout
```bash
# Add to torchrun command:
torchrun --nnodes=1 --nproc_per_node=6 --rdzv_timeout=3600 ...
```

## 🎨 After Training

### Visualize Results

```bash
# Copy trained checkpoint to model directory
rm -rf data/HunyuanVideo/transformer/*
cp data/outputs/grpo_hunyuan_6gpu/checkpoint-200-0/* data/HunyuanVideo/transformer/

# Generate test videos
bash scripts/visualization/vis_hunyuanvideo.sh
```

### Compare Before/After

Generate videos with the base model first:
```bash
# Before training (base model)
bash scripts/inference/inference_hunyuan_hf.sh

# After training (your checkpoint)
# (after copying checkpoint as shown above)
bash scripts/inference/inference_hunyuan_hf.sh
```

### Evaluate Quality

The reward curves show improvement:
```bash
# Plot rewards
python -c "
import matplotlib.pyplot as plt
rewards = [float(line.strip()) for line in open('vq_reward.txt')]
plt.plot(rewards)
plt.xlabel('Step')
plt.ylabel('VQ Reward')
plt.title('Video Quality Over Training')
plt.savefig('reward_curve.png')
"
```

## 📊 Configuration Details

### Modified Parameters Explained

**`--sp_size 2`**: Sequence parallel splits the temporal dimension
- 49 frames split across 2 GPUs = 25 frames per GPU
- Reduces memory by ~40% for long videos
- 6 GPUs → 3 parallel groups of 2

**`--use_cpu_offload`**: Offloads optimizer states to CPU RAM
- Saves ~10-15 GB GPU memory per GPU
- Minimal slowdown (~5-10%) on H100s

**`--selective_checkpointing 0.3`**: Checkpoints only 30% of layers
- Saves ~70% of activation memory
- Recomputes 70% during backward (trade compute for memory)

**`--gradient_accumulation_steps 8`**: Accumulates 8 micro-batches
- Effective batch size = 6 GPUs × 1 batch × 8 accum = 48
- Original: 32 GPUs × 1 batch × 4 accum = 128
- You have ~38% of original batch size (acceptable)

**`--timestep_fraction 0.5`**: Trains on 50% of denoising steps
- Randomly samples 6 out of 12 steps per update
- Saves compute, slight quality loss
- Can increase to 0.8 if training is fast enough

**`--num_generations 12`**: Generates 12 videos per prompt
- Original: 24 videos
- Best-of-N selects top 3 + bottom 3 for training
- Still enough diversity for learning

## 🔄 Scaling Up Later

If you get access to more GPUs:

**For 8 GPUs:**
```bash
--nproc_per_node=8
--h 416 --w 416 --t 49
--num_generations 16
--gradient_accumulation_steps 6
```

**For 16 GPUs:**
```bash
--nproc_per_node=16
--h 480 --w 480 --t 53
--num_generations 20
--gradient_accumulation_steps 4
--use_cpu_offload false  # No longer needed
```

## 💡 Tips for Best Results

1. **Start small**: Always test with minimal settings first
2. **Monitor memory**: Keep GPU memory under 70 GB to avoid OOM
3. **Check rewards frequently**: Should see improvement within 20-30 steps
4. **Save checkpoints**: Every 50 steps is good (takes ~30 seconds)
5. **Use TensorBoard/WandB**: Enable for better visualization
6. **Patience**: 3 hours is not long for video model training!

## 📚 Additional Resources

- **Original Paper**: [DanceGRPO: Unleashing GRPO on Visual Generation](https://arxiv.org/abs/2505.07818)
- **HunyuanVideo**: [Model Card](https://huggingface.co/hunyuanvideo-community/HunyuanVideo)
- **VideoAlign**: [Reward Model](https://huggingface.co/KwaiVGI/VideoReward)
- **Issues**: [GitHub Issues](https://github.com/XueZeyue/DanceGRPO/issues)

## ⚠️ Important Notes

1. **Text-to-Video only**: This config is for T2V. For Image-to-Video, use SkyReels-I2V instead.
2. **Preprocessing required**: You must preprocess text embeddings first (see Step 2).
3. **VideoAlign needed**: Download the reward model checkpoint before training.
4. **Disk space**: Need ~100 GB for checkpoints, videos, and logs.
5. **No resume yet**: Resuming from checkpoint is not implemented in the script.

---

**Ready to train?**
```bash
# Test first
bash scripts/finetune/finetune_hunyuan_grpo_6gpus_test.sh

# Then full training
bash scripts/finetune/finetune_hunyuan_grpo_6gpus.sh
```

Good luck! 🚀

