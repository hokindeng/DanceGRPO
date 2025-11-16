# Complete GRPO Training Flow: VideoAlign → HunyuanVideo

**A Detailed Walkthrough of How Reward Model Scores Update the Video Generation Model**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Step 1: Video Generation (Rollout)](#step-1-video-generation-rollout)
3. [Step 2: VideoAlign Reward Scoring](#step-2-videoalign-reward-scoring)
4. [Step 3: Reward to Advantage Conversion](#step-3-reward-to-advantage-conversion)
5. [Step 4: Best-of-N Selection](#step-4-best-of-n-selection)
6. [Step 5: PPO Loss Computation](#step-5-ppo-loss-computation)
7. [Step 6: Backpropagation](#step-6-backpropagation)
8. [Step 7: Weight Update](#step-7-weight-update)
9. [Complete Example End-to-End](#complete-example-end-to-end)
10. [Mathematical Deep Dive](#mathematical-deep-dive)

---

## Overview

**Goal:** Train HunyuanVideo to generate higher quality videos by using VideoAlign scores as learning signal.

**Key Insight:** We can't directly backpropagate through video pixels, so we use reinforcement learning:
- **Policy:** HunyuanVideo transformer (the model we're training)
- **Action:** Predicting noise/velocity at each denoising step
- **Reward:** VideoAlign score for the final video
- **Objective:** Maximize expected reward using PPO (Proximal Policy Optimization)

**Training Loop (One Iteration):**

```
Input: Text prompt "A cat walking in snow"
  ↓
Generate 8 videos with HunyuanVideo (different random seeds)
  ↓
Score each video with VideoAlign → [3.2, 4.1, 3.8, 2.9, 4.3, 3.5, 3.0, 3.7]
  ↓
Convert to advantages → [-0.375, 0.75, 0.375, -0.75, 1.0, 0.0, -0.625, 0.25]
  ↓
Select best 2 + worst 2 videos → Keep indices [4, 1, 0, 3]
  ↓
For each selected video, for each denoising timestep:
  - Recompute action probability with current policy
  - Compute PPO loss using advantage as weight
  - Backward() to get gradients
  ↓
Update HunyuanVideo weights with optimizer
  ↓
Next iteration: Model more likely to generate high-reward videos
```

---

## Step 1: Video Generation (Rollout)

**Purpose:** Generate multiple videos for the same prompt to explore different possibilities.

**Location:** `fastvideo/train_grpo_hunyuan.py`, function `sample_reference_model()` (lines 175-291)

### Main Loop

```python
def sample_reference_model(
    args,
    step,
    device, 
    transformer,      # HunyuanVideo model (in eval mode)
    vae,             # VAE decoder
    encoder_hidden_states,  # Text prompt embeddings
    encoder_attention_mask,
    inferencer,      # VideoAlign model
    caption,         # Text prompt strings
):
    # Put model in eval mode (no gradients, no dropout)
    transformer.eval()
    
    # Initialize lists to collect data
    all_latents = []      # Will store all intermediate denoising states
    all_log_probs = []    # Will store action probabilities
    all_vq_rewards = []   # Video quality scores
    all_mq_rewards = []   # Motion quality scores
    
    # Generate multiple videos for the same prompt
    for index in range(args.num_generations):  # e.g., 8 videos
        
        # === PHASE 1: DENOISING (Generate Video Latents) ===
        
        # Start from random noise
        z = torch.randn(
            (1, args.t, 16, latent_h, latent_w),  # Shape: [1, 45, 16, 40, 40]
            device=device, 
            dtype=torch.float32
        )
        
        # Initialize sigma schedule (noise levels)
        sigma_schedule = get_sigma_schedule(args.sampling_steps)
        # e.g., [14.61, 11.2, 8.3, 5.8, 3.7, 2.1, 1.0, 0.3, 0.0]
        
        latents_t = []      # Store states at each timestep
        log_probs_t = []    # Store log probabilities
        
        # Denoising loop: gradually remove noise
        for i, t in enumerate(timestep_schedule):
            
            # Forward pass through transformer (NO GRADIENTS)
            with torch.no_grad():
                model_pred = transformer(
                    hidden_states=z,                        # Current noisy latent
                    encoder_hidden_states=encoder_hidden_states,  # Text conditioning
                    timestep=torch.tensor([t]),             # Current timestep
                    guidance=torch.tensor([6018.0]),        # CFG scale
                    encoder_attention_mask=encoder_attention_mask,
                )[0]
            
            # Store current state
            latents_t.append(z)
            
            # Take one denoising step using predicted noise
            z, pred_original, log_prob = flux_step(
                model_output=model_pred,
                latents=z,
                eta=args.eta,           # Noise scale (0.25)
                sigmas=sigma_schedule,
                index=i,
                prev_sample=None,       # Sample new noise
                grpo=True,              # Compute log probability
                sde_solver=True
            )
            
            # Store log probability of taking this action
            log_probs_t.append(log_prob)
        
        # z is now the final denoised latent (shape: [1, 45, 16, 40, 40])
        
        # === PHASE 2: DECODE TO VIDEO ===
        
        with torch.no_grad():
            # Decode latent to RGB video frames
            video = vae.decode(z)  # Shape: [1, 45, 3, 320, 320]
            
            # Convert to uint8 numpy array
            videos = video_processor.postprocess_video(video)
            # Shape: [45, 320, 320, 3], dtype=uint8, range [0, 255]
        
        # === PHASE 3: SAVE VIDEO TO DISK ===
        
        rank = dist.get_rank()
        video_path = f"./videos/hunyuan_{rank}_{index}.mp4"
        export_to_video(videos[0], video_path, fps=args.fps)
        
        # === PHASE 4: GET VIDEOALIGN REWARD ===
        
        with torch.no_grad():
            absolute_path = os.path.abspath(video_path)
            
            # VideoAlign inference
            reward = inferencer.reward(
                [absolute_path],        # Video file path
                [caption[index]],       # Text prompt
                use_norm=True,          # Normalize using pre-computed stats
            )
            # reward[0] = {'VQ': 3.8, 'MQ': 0.5, 'TA': 0.2, 'Overall': 4.5}
            
            vq_reward = torch.tensor(reward[0]['VQ']).to(device)  # Video Quality
            mq_reward = torch.tensor(reward[0]['MQ']).to(device)  # Motion Quality
        
        # Store rewards
        all_vq_rewards.append(vq_reward.unsqueeze(0))
        all_mq_rewards.append(mq_reward.unsqueeze(0))
        
        # Store all latent states for this video
        all_latents.append(torch.stack(latents_t))
        all_log_probs.append(torch.stack(log_probs_t))
    
    # === CONCATENATE ALL VIDEOS ===
    
    all_latents = torch.cat(all_latents, dim=0)     # [8, T, 16, 40, 40]
    all_log_probs = torch.cat(all_log_probs, dim=0) # [8, T]
    all_vq_rewards = torch.cat(all_vq_rewards)      # [8]
    all_mq_rewards = torch.cat(all_mq_rewards)      # [8]
    
    return videos, z, all_vq_rewards, all_mq_rewards, all_latents, all_log_probs, sigma_schedule
```

### Denoising Step Details: `flux_step()`

**Location:** `fastvideo/train_grpo_hunyuan.py`, lines 57-96

```python
def flux_step(
    model_output: torch.Tensor,  # Predicted noise/velocity from transformer
    latents: torch.Tensor,       # Current state z_t
    eta: float,                  # Noise scale (0.25)
    sigmas: torch.Tensor,        # Noise schedule
    index: int,                  # Current step index
    prev_sample: torch.Tensor,   # If provided, compute log prob of this sample
    grpo: bool,                  # Whether to compute log probability
    sde_solver: bool,            # Use SDE formulation
):
    # Get current and next noise levels
    sigma = sigmas[index]           # e.g., σ_t = 5.8
    sigma_next = sigmas[index + 1]  # e.g., σ_{t+1} = 3.7
    dsigma = sigma_next - sigma     # e.g., -2.1
    
    # Predicted next state (mean of distribution)
    prev_sample_mean = latents + dsigma * model_output
    # μ_{t+1} = z_t + (σ_{t+1} - σ_t) × v_θ(z_t, t, text)
    
    # Predicted clean sample (denoised)
    pred_original_sample = latents - sigma * model_output
    # x_0 = z_t - σ_t × v_θ(z_t, t, text)
    
    # Compute noise standard deviation
    delta_t = sigma - sigma_next  # e.g., 2.1
    std_dev_t = eta * math.sqrt(delta_t)  # σ_noise = 0.25 × √2.1 ≈ 0.362
    
    # SDE correction term (optional)
    if sde_solver:
        score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma
    
    # Sample next state if not provided
    if grpo and prev_sample is None:
        noise = torch.randn_like(prev_sample_mean)
        prev_sample = prev_sample_mean + noise * std_dev_t
        # z_{t+1} ~ N(μ_{t+1}, σ_noise^2)
    
    # Compute log probability if in GRPO mode
    if grpo:
        # Log probability under Gaussian distribution:
        # log p(z_{t+1} | z_t, text) = log N(z_{t+1}; μ_{t+1}, σ_noise^2)
        #                            = -||z_{t+1} - μ_{t+1}||^2 / (2σ_noise^2) 
        #                              - log(σ_noise) - log(√(2π))
        
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t ** 2))
            - math.log(std_dev_t) 
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        
        # Average over all dimensions except batch
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        # Shape: [batch_size]
        
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean, pred_original_sample
```

**Key Insight:** The `log_prob` tells us how likely the model was to take this specific denoising action. This is crucial for reinforcement learning!

---

## Step 2: VideoAlign Reward Scoring

**Purpose:** Evaluate video quality using a trained vision-language model.

**Location:** `fastvideo/models/videoalign/inference.py`

### VideoAlign Architecture

VideoAlign is a **Qwen2-VL model** fine-tuned to predict video quality scores.

```python
class VideoVLMRewardInference():
    def __init__(self, load_from_pretrained, device='cuda', dtype=torch.bfloat16):
        # Load configuration
        config_path = os.path.join(load_from_pretrained, "model_config.json")
        data_config, model_config, peft_lora_config = load_configs_from_json(config_path)
        
        # Load Qwen2-VL model with reward head
        model, processor, peft_config = create_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
        )
        
        # Load checkpoint weights
        model, checkpoint_step = load_model_from_checkpoint(
            model, load_from_pretrained, load_from_pretrained_step=-1
        )
        
        model.eval()
        self.model = model.to(device)
        self.processor = processor
    
    def reward(self, video_paths, prompts, fps=None, num_frames=None, use_norm=True):
        """
        Compute rewards for videos given text prompts.
        
        Args:
            video_paths: List[str], paths to video files
            prompts: List[str], text descriptions
            use_norm: bool, whether to normalize using dataset statistics
            
        Returns:
            List[dict], each with keys 'VQ', 'MQ', 'TA', 'Overall'
        """
        # Prepare video + text input
        batch = self.prepare_batch(video_paths, prompts, fps, num_frames)
        # batch = {
        #     'input_ids': tensor([[1, 2, 3, ...]]),        # Tokenized text
        #     'attention_mask': tensor([[1, 1, 1, ...]]),
        #     'pixel_values': tensor([[[...]]])             # Video frames
        # }
        
        # Forward pass through Qwen2-VL
        with torch.no_grad():
            outputs = self.model(return_dict=True, **batch)
            logits = outputs["logits"]  # Shape: [batch_size, 3]
            # 3 dimensions: [VQ, MQ, TA]
        
        # Convert logits to reward dictionary
        rewards = []
        for i, logit in enumerate(logits):
            reward = {
                'VQ': logit[0].item(),  # Video Quality score
                'MQ': logit[1].item(),  # Motion Quality score  
                'TA': logit[2].item(),  # Text Alignment score
            }
            
            # Normalize using pre-computed statistics
            if use_norm:
                reward['VQ'] = (reward['VQ'] - self.vq_mean) / self.vq_std
                reward['MQ'] = (reward['MQ'] - self.mq_mean) / self.mq_std
                reward['TA'] = (reward['TA'] - self.ta_mean) / self.ta_std
            
            # Compute overall score
            reward['Overall'] = reward['VQ'] + reward['MQ'] + reward['TA']
            rewards.append(reward)
        
        return rewards
```

### Example Output

```python
# Input
video_path = "./videos/hunyuan_0_0.mp4"
prompt = "A cat walking in snow"

# VideoAlign inference
reward = inferencer.reward([video_path], [prompt], use_norm=True)

# Output
reward[0] = {
    'VQ': 3.8,      # Above average video quality
    'MQ': 0.5,      # Average motion quality
    'TA': 0.2,      # Slightly above average text alignment
    'Overall': 4.5  # Sum of all three
}
```

**Key Properties:**
- **VQ (Video Quality):** Measures visual clarity, artifacts, aesthetics
- **MQ (Motion Quality):** Measures smoothness, realism of motion
- **TA (Text Alignment):** Measures how well video matches prompt
- **Normalized scores:** Mean ≈ 0, Std ≈ 1 (from dataset statistics)

---

## Step 3: Reward to Advantage Conversion

**Purpose:** Normalize rewards within each generation group to focus on relative quality.

**Location:** `fastvideo/train_grpo_hunyuan.py`, `train_one_step()` function, lines 363-385

### Why Advantages?

**Problem:** Absolute rewards vary a lot:
- Different prompts have different difficulty levels
- Some prompts naturally produce higher/lower scores
- We care about **which videos are better** for a given prompt

**Solution:** Convert to **advantages** = normalized rewards within group

### Code

```python
def train_one_step(args, device, transformer, vae, inferencer, 
                   optimizer, lr_scheduler, loader, max_grad_norm, step):
    
    # ... (generate videos, get rewards) ...
    
    # samples["vq_rewards"] shape: [8] (8 videos for 1 prompt)
    # Example: [3.2, 4.1, 3.8, 2.9, 4.3, 3.5, 3.0, 3.7]
    
    n = len(samples["vq_rewards"]) // args.num_generations  # n = 1 (one prompt)
    vq_advantages = torch.zeros_like(samples["vq_rewards"])
    
    # Process each prompt's generation group
    for i in range(n):
        start_idx = i * args.num_generations  # 0
        end_idx = (i + 1) * args.num_generations  # 8
        
        # Get all rewards for this prompt
        group_rewards = samples["vq_rewards"][start_idx:end_idx]
        # [3.2, 4.1, 3.8, 2.9, 4.3, 3.5, 3.0, 3.7]
        
        # Compute group statistics
        group_mean = group_rewards.mean()  # μ = 3.5625
        group_std = group_rewards.std() + 1e-8  # σ = 0.4798 (+ epsilon for stability)
        
        # Z-score normalization: (x - μ) / σ
        vq_advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
        # [
        #   (3.2 - 3.5625) / 0.4798 = -0.756,
        #   (4.1 - 3.5625) / 0.4798 =  1.121,
        #   (3.8 - 3.5625) / 0.4798 =  0.495,
        #   (2.9 - 3.5625) / 0.4798 = -1.381,
        #   (4.3 - 3.5625) / 0.4798 =  1.538,  ← Best video
        #   (3.5 - 3.5625) / 0.4798 = -0.130,
        #   (3.0 - 3.5625) / 0.4798 = -1.172,
        #   (3.7 - 3.5625) / 0.4798 =  0.286
        # ]
    
    samples["vq_advantages"] = vq_advantages
    
    # Repeat for motion quality
    mq_advantages = torch.zeros_like(samples["mq_rewards"])
    for i in range(n):
        start_idx = i * args.num_generations
        end_idx = (i + 1) * args.num_generations
        group_rewards = samples["mq_rewards"][start_idx:end_idx]
        group_mean = group_rewards.mean()
        group_std = group_rewards.std() + 1e-8
        mq_advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
    
    samples["mq_advantages"] = mq_advantages
```

### Interpretation

**Advantages after normalization:**
```
Video 0: advantage = -0.756  → Below average (make less likely)
Video 1: advantage =  1.121  → Above average (make more likely)
Video 2: advantage =  0.495  → Slightly above average
Video 3: advantage = -1.381  → Worst video (make much less likely)
Video 4: advantage =  1.538  → Best video (make much more likely)
Video 5: advantage = -0.130  → Slightly below average
Video 6: advantage = -1.172  → Second worst
Video 7: advantage =  0.286  → Slightly above average
```

**Key Properties:**
- **Mean = 0:** Advantages are balanced (sum to ~0)
- **Std = 1:** Unit variance makes them comparable
- **Positive = Better than average** for this prompt
- **Negative = Worse than average** for this prompt

---

## Step 4: Best-of-N Selection

**Purpose:** Focus training on the most informative samples (best and worst).

**Location:** `fastvideo/train_grpo_hunyuan.py`, lines 388-402

### Motivation

Training on **all 8 videos** is wasteful:
- Middle-quality videos (advantage ≈ 0) provide weak learning signal
- Best and worst videos provide **strongest signal**
- Reduces computation and memory

### Code

```python
# Combine VQ and MQ advantages with coefficients
total_scores = args.vq_coef * samples["vq_advantages"] + args.mq_coef * samples["mq_advantages"]
# With vq_coef=1.0, mq_coef=0.0:
# total_scores = vq_advantages
# = [-0.756, 1.121, 0.495, -1.381, 1.538, -0.130, -1.172, 0.286]

# Sort by score
sorted_indices = torch.argsort(total_scores)
# sorted_indices = [3, 6, 0, 5, 7, 2, 1, 4]
#                   ↑worst              ↑best

# Select top N/2 and bottom N/2
top_indices = sorted_indices[-args.bestofn//2:]    # Top 2
# top_indices = [1, 4]  (advantages: 1.121, 1.538)

bottom_indices = sorted_indices[:args.bestofn//2]  # Bottom 2
# bottom_indices = [3, 6]  (advantages: -1.381, -1.172)

# Concatenate and shuffle
selected_indices = torch.cat([top_indices, bottom_indices])
# selected_indices = [1, 4, 3, 6]

shuffled_order = torch.randperm(len(selected_indices))
# shuffled_order = [2, 0, 3, 1]

selected_indices = selected_indices[shuffled_order]
# selected_indices = [3, 1, 6, 4]  (randomized order)

# Filter all sample data to keep only selected videos
if args.num_generations != args.bestofn:  # 8 != 4
    for key in samples:
        samples[key] = samples[key][selected_indices]
    batch_size = len(selected_indices)  # Now 4 instead of 8

# Now samples contains only 4 videos:
# samples["vq_advantages"] = [-1.381, 1.121, -1.172, 1.538]
# samples["latents"].shape = [4, T, 16, 40, 40]  (was [8, T, 16, 40, 40])
```

### Result

**Before:** Training on 8 videos × T timesteps = 8T updates
**After:** Training on 4 videos × T timesteps = 4T updates

**Selected videos:**
1. Video 3 (advantage = -1.381) ← Worst, learn to avoid
2. Video 1 (advantage = 1.121) ← Good, learn to replicate
3. Video 6 (advantage = -1.172) ← Second worst, learn to avoid
4. Video 4 (advantage = 1.538) ← Best, learn to replicate

---

## Step 5: PPO Loss Computation

**Purpose:** Compute the training loss that will drive gradient descent.

**Location:** `fastvideo/train_grpo_hunyuan.py`, lines 424-470

### Setup: Timestep Shuffling

```python
# Randomize timestep order for each video (reduces correlation)
perms = torch.stack([
    torch.randperm(len(samples["timesteps"][0]))  # Random permutation for each video
    for _ in range(batch_size)
]).to(device)
# perms shape: [4, T]
# Example: perms[0] = [3, 7, 1, 9, 0, 5, 2, 8, 4, 6]

# Reorder timesteps, latents, etc. according to permutation
for key in ["timesteps", "latents", "next_latents", "log_probs"]:
    samples[key] = samples[key][
        torch.arange(batch_size)[:, None],
        perms,
    ]

# Reshape for easier iteration
samples_batched = {k: v.unsqueeze(1) for k, v in samples.items()}
samples_batched_list = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]
# Now samples_batched_list is a list of 4 dicts, one per video
```

### Main Training Loop

```python
train_timesteps = int(len(samples["timesteps"][0]) * args.timestep_fraction)
# e.g., 10 timesteps × 0.5 = 5 timesteps to train on

total_loss = 0.0

# Iterate over selected videos
for i, sample in enumerate(samples_batched_list):  # 4 videos
    
    # Iterate over denoising timesteps
    for _ in range(train_timesteps):  # 5 timesteps
        
        clip_range = 1e-4    # PPO clipping epsilon
        adv_clip_max = 5.0   # Maximum advantage magnitude
        
        # ==========================================
        # STEP 5A: Recompute log probability
        # ==========================================
        
        new_log_probs = grpo_one_step(
            args,
            sample["latents"][:,_],            # z_t (current state)
            sample["next_latents"][:,_],       # z_{t+1} (next state from rollout)
            sample["encoder_hidden_states"],   # Text embedding
            sample["encoder_attention_mask"],
            transformer,                       # HunyuanVideo (in train mode!)
            sample["timesteps"][:,_],          # Current timestep
            perms[i][_],                       # Timestep index
            sigma_schedule,
        )
        # new_log_probs shape: [1] (scalar per video)
        # Example: new_log_probs = tensor([-5.2341])
        
        # ==========================================
        # STEP 5B: Compute importance sampling ratio
        # ==========================================
        
        old_log_probs = sample["log_probs"][:,_]
        # old_log_probs from rollout: tensor([-5.0123])
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        # ratio = exp(-5.2341 - (-5.0123)) = exp(-0.2218) = 0.8010
        
        # ==========================================
        # STEP 5C: Compute PPO loss for VQ
        # ==========================================
        
        # Clip advantages to prevent extreme values
        vq_advantages = torch.clamp(
            sample["vq_advantages"],
            -adv_clip_max,  # -5.0
            adv_clip_max,   # +5.0
        )
        # vq_advantages = tensor([1.538])  (for video 4, the best)
        
        # Unclipped loss: L = -A × r
        vq_unclipped_loss = -vq_advantages * ratio
        # = -1.538 × 0.8010 = -1.2319
        
        # Clipped loss: L_clip = -A × clip(r, 1-ε, 1+ε)
        vq_clipped_loss = -vq_advantages * torch.clamp(
            ratio,
            1.0 - clip_range,  # 0.9999
            1.0 + clip_range,  # 1.0001
        )
        # ratio = 0.8010 is clamped to 0.9999
        # = -1.538 × 0.9999 = -1.5378
        
        # Take maximum (least negative) of the two
        vq_loss = torch.mean(torch.maximum(vq_unclipped_loss, vq_clipped_loss))
        # = mean(max(-1.2319, -1.5378)) = -1.2319
        
        # Normalize by accumulation steps and timesteps
        vq_loss = vq_loss / (args.gradient_accumulation_steps * train_timesteps)
        # = -1.2319 / (12 × 5) = -0.02053
        
        # ==========================================
        # STEP 5D: Compute PPO loss for MQ (same formula)
        # ==========================================
        
        mq_advantages = torch.clamp(sample["mq_advantages"], -adv_clip_max, adv_clip_max)
        mq_unclipped_loss = -mq_advantages * ratio
        mq_clipped_loss = -mq_advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        mq_loss = torch.mean(torch.maximum(mq_unclipped_loss, mq_clipped_loss))
        mq_loss = mq_loss / (args.gradient_accumulation_steps * train_timesteps)
        
        # ==========================================
        # STEP 5E: Combine losses
        # ==========================================
        
        final_loss = args.vq_coef * vq_loss + args.mq_coef * mq_loss
        # = 1.0 × (-0.02053) + 0.0 × mq_loss = -0.02053
        
        # ... (next step: backward) ...
```

### `grpo_one_step()` - Detailed Breakdown

**Location:** Lines 146-173

```python
def grpo_one_step(
    args,
    latents,              # Current state z_t, shape: [1, 16, 40, 40]
    pre_latents,          # Next state z_{t+1} from rollout
    encoder_hidden_states,  # Text embedding, shape: [1, L, D]
    encoder_attention_mask,
    transformer,          # HunyuanVideo model
    timesteps,            # Current timestep value, shape: [1]
    i,                    # Timestep index
    sigma_schedule,       # Noise schedule
):
    B = encoder_hidden_states.shape[0]  # Batch size = 1
    
    # Forward pass with mixed precision
    with torch.autocast("cuda", torch.bfloat16):
        # PUT MODEL IN TRAINING MODE (enables gradients!)
        transformer.train()
        
        # FORWARD PASS THROUGH TRANSFORMER
        model_pred = transformer(
            hidden_states=latents,                # z_t
            encoder_hidden_states=encoder_hidden_states,  # text
            timestep=timesteps,                   # t
            guidance=torch.tensor([6018.0], device=latents.device, dtype=torch.bfloat16),
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )[0]
        # model_pred shape: [1, 16, 40, 40]
        # This is v_θ(z_t, t, text) - the predicted velocity
    
    # Compute log probability using flux_step
    z, pred_original, log_prob = flux_step(
        model_output=model_pred,
        latents=latents.to(torch.float32),
        eta=args.eta,
        sigmas=sigma_schedule,
        index=i,
        prev_sample=pre_latents.to(torch.float32),  # The actual z_{t+1} we sampled
        grpo=True,
        sde_solver=True
    )
    
    # log_prob = log p_θ(z_{t+1} | z_t, text)
    # This is the probability under the CURRENT policy θ
    # of taking the action that leads to z_{t+1}
    
    return log_prob  # Shape: [1]
```

### PPO Clipping Explained

**Why clip the ratio?**

Without clipping, if the model changes drastically:
- `ratio >> 1`: New policy much more likely to take action than old
- Could lead to overly aggressive updates and instability

**PPO solution:**
```python
ratio_clipped = clamp(ratio, 1 - ε, 1 + ε)  # ε = 0.0001
loss = max(L_unclipped, L_clipped)
```

**Case Analysis:**

| Scenario | Advantage | Ratio | Unclipped Loss | Clipped Loss | Final Loss | Effect |
|----------|-----------|-------|----------------|--------------|------------|--------|
| Good video, policy improves | +1.5 | 1.2 | -1.8 | -1.5002 | -1.5002 | Clip prevents over-optimization |
| Good video, policy degrades | +1.5 | 0.8 | -1.2 | -1.4999 | -1.2 | Allow improvement |
| Bad video, policy improves | -1.5 | 0.8 | 1.2 | 1.4999 | 1.4999 | Clip prevents over-penalization |
| Bad video, policy degrades | -1.5 | 1.2 | 1.8 | 1.5002 | 1.8 | Allow penalty |

**Key insight:** Clipping ensures we don't update too aggressively in the "good" direction, but allows reasonable updates.

---

## Step 6: Backpropagation

**Purpose:** Compute gradients of the loss with respect to all transformer parameters.

**Location:** `fastvideo/train_grpo_hunyuan.py`, line 472

### The Magic Line

```python
final_loss.backward()
```

**What happens internally:**

1. **PyTorch's autograd engine traces the computation graph:**
```
final_loss
  ↑
vq_loss (+ mq_loss)
  ↑
torch.maximum(unclipped_loss, clipped_loss)
  ↑
-vq_advantages × ratio (or clipped_ratio)
  ↑
exp(new_log_probs - old_log_probs)
  ↑
new_log_probs = grpo_one_step(...)
  ↑
flux_step(..., model_pred, ...)
  ↑
model_pred = transformer(latents, text, timestep)
  ↑
All transformer layers, attention, MLP, etc.
  ↑
transformer.parameters() ← 13.5 billion weights!
```

2. **Apply chain rule to compute gradients:**

For each parameter `w` in the transformer:
```
∂(final_loss)/∂w = ∂(final_loss)/∂(vq_loss) × 
                   ∂(vq_loss)/∂(ratio) × 
                   ∂(ratio)/∂(new_log_probs) × 
                   ∂(new_log_probs)/∂(model_pred) × 
                   ∂(model_pred)/∂w
```

3. **Store gradients in `.grad` attributes:**

```python
# Example: Attention layer query weights
transformer.layers[0].attn.query.weight.grad
# Shape: [4096, 4096]
# Example values: [[-0.0001, 0.0003, -0.0002, ...], ...]

# Example: Output projection weights
transformer.final_layer.weight.grad
# Shape: [16, 4096]
# Example values: [[0.0005, -0.0001, 0.0002, ...], ...]
```

### Gradient Synchronization Across GPUs

```python
# Line 473-475: Average loss across all GPUs
avg_loss = final_loss.detach().clone()
dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
total_loss += avg_loss.item()

# Gradients are automatically synchronized by FSDP
# Each GPU computes gradients for its shard
# FSDP all-reduces gradients during backward pass
```

### Memory During Backward

**Forward pass:** Activations stored in memory (for gradient computation)
**Backward pass:** 
- Activations loaded from memory (or recomputed if using gradient checkpointing)
- Gradients computed and accumulated
- Activations deallocated after use

**With gradient checkpointing (enabled in config):**
- Only 20% of activations stored (selective_checkpointing=0.2)
- Other 80% recomputed during backward
- Trade: More compute, less memory

---

## Step 7: Weight Update

**Purpose:** Apply the computed gradients to update transformer parameters.

**Location:** `fastvideo/train_grpo_hunyuan.py`, lines 476-480

### Gradient Accumulation Check

```python
# Line 476: Only update after accumulating enough gradients
if (i+1) % args.gradient_accumulation_steps == 0:  # Every 12 samples
```

**Why accumulate?**
- Simulate larger batch size without OOM
- More stable gradients (average over 12 samples)
- Gradient accumulation = 12 means effective batch size = 6 GPUs × 1 × 12 = 72

### Gradient Clipping

```python
# Line 477: Prevent gradient explosion
grad_norm = transformer.clip_grad_norm_(max_grad_norm)

# What this does:
# 1. Compute total gradient norm across all parameters:
#    total_norm = sqrt(sum(||grad_i||^2 for all parameters))
#
# 2. If total_norm > max_grad_norm (1.0):
#    scale_factor = max_grad_norm / total_norm
#    for param in transformer.parameters():
#        param.grad *= scale_factor
#
# Example:
# total_norm = 2.5
# max_grad_norm = 1.0
# scale_factor = 1.0 / 2.5 = 0.4
# All gradients multiplied by 0.4
```

**Why clip?**
- Large gradients can cause unstable training
- Especially important in RL where reward signal can be noisy
- Ensures no single update is too large

### The Update Step

```python
# Line 478: UPDATE WEIGHTS!
optimizer.step()
```

**What happens inside `optimizer.step()` (AdamW):**

For each parameter `w`:

```python
# AdamW update rule:

# 1. Get gradient
g = w.grad

# 2. Update first moment (exponential moving average of gradients)
m = beta1 * m + (1 - beta1) * g  # beta1 = 0.9
# m is the "momentum" - smoothed gradient direction

# 3. Update second moment (exponential moving average of squared gradients)
v = beta2 * v + (1 - beta2) * g^2  # beta2 = 0.999
# v tracks the variance of gradients

# 4. Bias correction (especially important in early training)
m_hat = m / (1 - beta1^t)  # t = step number
v_hat = v / (1 - beta2^t)

# 5. Adaptive learning rate for this parameter
adaptive_lr = learning_rate / (sqrt(v_hat) + epsilon)
# Parameters with large, consistent gradients get smaller effective LR
# Parameters with small, noisy gradients get larger effective LR

# 6. Weight decay (L2 regularization, applied directly to weights)
w = w * (1 - learning_rate * weight_decay)
# = w * (1 - 8e-6 * 0.0001) = w * 0.9999999992

# 7. FINAL UPDATE
w_new = w - adaptive_lr * m_hat
```

**Concrete example:**

```python
# Parameter: transformer.layers[10].attn.query.weight[0, 0]
w_old = 0.123456

# Gradient after backprop
grad = -0.0001234

# AdamW state (after many steps)
m = -0.0000987  # Smoothed gradient
v = 0.0000000152  # Smoothed squared gradient

# Bias correction (assume step 1000)
m_hat = m / (1 - 0.9^1000) ≈ m  # ≈ -0.0000987
v_hat = v / (1 - 0.999^1000) ≈ v  # ≈ 0.0000000152

# Adaptive learning rate
adaptive_lr = 8e-6 / (sqrt(0.0000000152) + 1e-8)
            = 8e-6 / (0.0001233 + 1e-8)
            = 0.0649

# Weight decay
w_old = 0.123456 * (1 - 8e-6 * 0.0001)
      = 0.123456 * 0.9999999992
      ≈ 0.123456  # Negligible

# Final update
w_new = 0.123456 - 0.0649 * (-0.0000987)
      = 0.123456 + 0.0000064
      = 0.1234624

# Change: +0.0000064 (tiny increase because gradient is negative)
```

**Interpretation:**
- Gradient = -0.0001234 (negative) → Loss decreases when weight increases
- Update: Weight increases slightly
- Effect: Model becomes slightly more likely to predict this action
- For a good video (positive advantage), this is what we want!

### Learning Rate Scheduling

```python
# Line 479: Update learning rate schedule
lr_scheduler.step()
```

**Schedule used:** `constant_with_warmup`

```python
# Warmup phase (first 10 steps):
if step < warmup_steps:
    lr = base_lr * (step / warmup_steps)
    # Step 1: lr = 8e-6 * (1/10) = 8e-7
    # Step 5: lr = 8e-6 * (5/10) = 4e-6
    # Step 10: lr = 8e-6 * (10/10) = 8e-6
else:
    lr = base_lr  # 8e-6 (constant)
```

**Why warmup?**
- Large gradients early in training can destabilize
- Gradual warmup allows model to "settle in"
- After warmup, constant LR is sufficient

### Zero Gradients

```python
# Line 480: Clear gradients for next accumulation cycle
optimizer.zero_grad()

# What this does:
for param in transformer.parameters():
    param.grad = None  # or fill with zeros
```

**Why?**
- PyTorch accumulates gradients by default (`grad += new_grad`)
- Must clear after each optimizer step
- Otherwise, gradients from previous batches would accumulate

---

## Complete Example End-to-End

Let's trace through **one complete training iteration** with concrete numbers.

### Initial Setup

```python
# Configuration
prompt = "A cat walking in snow"
num_generations = 8
bestofn = 4
learning_rate = 8e-6
gradient_accumulation_steps = 12
```

### Step 1: Generate 8 Videos

```python
# Generate 8 videos with different random seeds
for i in range(8):
    z = torch.randn(...)  # Different random noise each time
    video_i = generate_video(z, prompt, transformer)
    save_video(f"video_{i}.mp4", video_i)
```

### Step 2: VideoAlign Scoring

```python
# Score each video
vq_rewards = []
for i in range(8):
    reward = videoalign.reward(f"video_{i}.mp4", prompt)
    vq_rewards.append(reward['VQ'])

# Results
vq_rewards = torch.tensor([3.2, 4.1, 3.8, 2.9, 4.3, 3.5, 3.0, 3.7])
```

### Step 3: Compute Advantages

```python
mean = vq_rewards.mean()  # 3.5625
std = vq_rewards.std()    # 0.4798

advantages = (vq_rewards - mean) / std
# = [-0.756, 1.121, 0.495, -1.381, 1.538, -0.130, -1.172, 0.286]
```

### Step 4: Best-of-N Selection

```python
sorted_indices = torch.argsort(advantages)
# [3, 6, 0, 5, 7, 2, 1, 4]

# Select best 2 and worst 2
selected = [4, 1, 3, 6]  # Indices
selected_advantages = [1.538, 1.121, -1.381, -1.172]
```

### Step 5-7: Train on Selected Videos

Train on video 4 (best, advantage = 1.538):

```python
# === Timestep 0 ===

# Forward pass
new_log_prob = grpo_one_step(latents_t0, latents_t1, prompt, transformer)
# new_log_prob = -5.234

# Importance sampling ratio
old_log_prob = -5.012  # From rollout
ratio = exp(new_log_prob - old_log_prob)
# ratio = exp(-5.234 - (-5.012)) = exp(-0.222) = 0.801

# PPO loss
advantage = 1.538
unclipped_loss = -advantage * ratio
# = -1.538 * 0.801 = -1.232

clipped_loss = -advantage * clamp(ratio, 0.9999, 1.0001)
# = -1.538 * 0.9999 = -1.538

vq_loss = max(unclipped_loss, clipped_loss) / (12 * 5)
# = max(-1.232, -1.538) / 60 = -1.232 / 60 = -0.0205

# Backprop
vq_loss.backward()  # Accumulate gradients

# === Timestep 1 ===
# (repeat 4 more times for timesteps 1-4)

# === After 5 timesteps ===
# Total accumulated gradient in each parameter

# === After processing all 4 selected videos ===
# (4 videos × 5 timesteps = 20 backward passes)

# === Gradient accumulation step complete ===

# Clip gradients
total_norm = sqrt(sum(grad^2))  # e.g., 0.823
# total_norm < 1.0, so no clipping needed

# Update weights
optimizer.step()

# Example weight update:
# w_old = 0.1234567
# grad = -0.000123 (accumulated over 20 backward passes)
# m = 0.9 * m_prev + 0.1 * grad = -0.000098
# v = 0.999 * v_prev + 0.001 * grad^2 = 0.000000015
# adaptive_lr = 8e-6 / sqrt(v) = 0.0649
# w_new = w_old - adaptive_lr * m
#       = 0.1234567 - 0.0649 * (-0.000098)
#       = 0.1234567 + 0.0000064
#       = 0.1234631

# Clear gradients
optimizer.zero_grad()
```

### Result After This Iteration

- Transformer weights updated by tiny amounts
- Parameters associated with generating video 4 (best) → increased
- Parameters associated with generating video 3 (worst) → decreased
- Next iteration: Model slightly more likely to generate high-quality videos

### After 200 Iterations

- Cumulative effect of thousands of small updates
- Model learns:
  - Better composition (from VQ scores)
  - More realistic motion (from MQ scores)
  - Better prompt following (from TA scores)
- Expected reward improvement: 10-30% over initial policy

---

## Mathematical Deep Dive

### Reinforcement Learning Formulation

**MDP (Markov Decision Process):**
- **State** `s_t = z_t`: Latent at timestep t
- **Action** `a_t`: Sampling noise to go from z_t to z_{t+1}
- **Policy** `π_θ(a_t | s_t, text)`: Probability distribution over actions (Gaussian)
- **Reward** `R`: VideoAlign score (received only at the end)

**Objective:** Maximize expected reward
```
J(θ) = E_{τ ~ π_θ} [R(τ)]
```
where τ = (s_0, a_0, s_1, a_1, ..., s_T) is a trajectory (full video generation)

### Policy Gradient Theorem

```
∇_θ J(θ) = E_{τ ~ π_θ} [∑_t ∇_θ log π_θ(a_t | s_t, text) × A_t]
```

where `A_t` is the advantage function.

**Intuition:**
- `∇_θ log π_θ`: Direction to change θ to make action a_t more likely
- `A_t`: How much better this action was than average
- Product: Push probability up for good actions, down for bad actions

**In our code:**
```python
# ∇_θ log π_θ(a_t | s_t) is computed by:
log_prob = flux_step(...)  # log π_θ(a_t | s_t, text)
loss = -advantage * ratio   # -A_t × exp(log π_θ - log π_θ_old)
loss.backward()             # Computes ∇_θ loss = ∇_θ [-A × π_θ/π_θ_old]
                            # ≈ ∇_θ [-A × π_θ] / π_θ_old
                            # ≈ -A × ∇_θ π_θ / π_θ_old
                            # = -A × π_θ × ∇_θ log π_θ / π_θ_old (log derivative trick)
```

### PPO Objective

**Problem with vanilla policy gradient:**
- Can take too large steps, causing policy collapse
- If π_θ becomes very different from π_θ_old, importance sampling breaks down

**PPO solution:** Clip the importance sampling ratio

```
L_PPO(θ) = E [min(
    r(θ) × A,                    # Unclipped
    clip(r(θ), 1-ε, 1+ε) × A     # Clipped
)]

where r(θ) = π_θ(a|s) / π_θ_old(a|s)
```

**Clipping ensures:**
- When A > 0 (good action):
  - If r > 1+ε (new policy much more likely): Clip to 1+ε
  - Prevents over-optimization
- When A < 0 (bad action):
  - If r < 1-ε (new policy much less likely): Clip to 1-ε
  - Prevents over-penalization

**In our code:**
```python
ratio = torch.exp(new_log_probs - old_log_probs)  # r(θ)
unclipped_loss = -vq_advantages * ratio
clipped_loss = -vq_advantages * torch.clamp(ratio, 1-1e-4, 1+1e-4)
vq_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
```

We take `max()` instead of `min()` because we have a **negative sign** (minimizing loss = maximizing objective).

### Gaussian Policy

**Denoising step as a stochastic policy:**

```
z_{t+1} ~ N(μ_θ(z_t, text, t), σ^2)

where μ_θ = z_t + (σ_{t+1} - σ_t) × v_θ(z_t, text, t)
```

**Log probability:**
```
log π_θ(z_{t+1} | z_t, text) = log N(z_{t+1}; μ_θ, σ^2)
                              = -||z_{t+1} - μ_θ||^2 / (2σ^2) - log σ - log √(2π)
```

**In our code (flux_step):**
```python
prev_sample_mean = latents + dsigma * model_output  # μ_θ
std_dev_t = eta * math.sqrt(delta_t)                # σ

log_prob = (
    -((prev_sample - prev_sample_mean) ** 2) / (2 * (std_dev_t ** 2))
    - math.log(std_dev_t)
    - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
)
```

### Advantage Normalization

**Why normalize advantages within each group?**

**Without normalization:**
```
Prompt A: rewards = [3.0, 3.5, 4.0, 4.5]  (easy prompt)
Prompt B: rewards = [2.0, 2.5, 3.0, 3.5]  (hard prompt)

Direct use: Prompt A videos always get higher weight
Problem: Model learns to prefer certain prompts, not improve quality
```

**With normalization:**
```
Prompt A: advantages = [-1.34, -0.45, 0.45, 1.34]
Prompt B: advantages = [-1.34, -0.45, 0.45, 1.34]

Effect: Focus on relative quality within each prompt
Result: Model learns to improve quality regardless of prompt difficulty
```

**Code:**
```python
group_mean = rewards[start:end].mean()
group_std = rewards[start:end].std() + 1e-8
advantages = (rewards - group_mean) / group_std
```

This is **crucial** for multi-prompt training!

---

## Summary

**The complete flow:**

1. **Generate** 8 videos with HunyuanVideo for prompt "A cat walking in snow"
2. **Score** each video with VideoAlign → [3.2, 4.1, 3.8, 2.9, 4.3, 3.5, 3.0, 3.7]
3. **Normalize** to advantages → [-0.76, 1.12, 0.50, -1.38, 1.54, -0.13, -1.17, 0.29]
4. **Select** best 2 + worst 2 → Keep videos [4, 1, 3, 6]
5. For each selected video, for each denoising timestep:
   - **Forward** through transformer to get new action probability
   - **Compute** PPO loss = -advantage × ratio (with clipping)
   - **Backward** to accumulate gradients
6. **Clip** gradients to prevent explosion
7. **Update** transformer weights using AdamW
8. **Repeat** for 200+ iterations

**Result:** HunyuanVideo learns to generate higher quality videos that score better on VideoAlign!

**Key insights:**
- RL treats video generation as a sequential decision process
- PPO provides stable training with clipping
- Advantage normalization focuses on relative quality
- Best-of-N selection concentrates learning on most informative samples
- Tiny weight updates accumulate over iterations to meaningful improvement

---

## Code File Reference

**Main training script:**
- `fastvideo/train_grpo_hunyuan.py`
  - `sample_reference_model()`: Lines 175-291 (video generation + reward)
  - `train_one_step()`: Lines 298-492 (complete training step)
  - `grpo_one_step()`: Lines 146-173 (transformer forward + log prob)
  - `flux_step()`: Lines 57-96 (denoising step + Gaussian log prob)

**VideoAlign reward model:**
- `fastvideo/models/videoalign/inference.py`
  - `VideoVLMRewardInference.reward()`: Lines 176-203 (reward computation)

**Training config:**
- `scripts/finetune/finetune_hunyuan_grpo_6gpus.sh` (all hyperparameters)

---

**This document provides a complete picture of how reinforcement learning with human preferences (via VideoAlign) trains a diffusion model (HunyuanVideo) to generate better videos!**

