# GRPO for Video Models - Flow Diagram

## Main Training Loop

```mermaid
flowchart TD
    Start([Start Training]) --> LoadData[Load Prompt + Text Embeddings]
    LoadData --> GroupGen{Use Group Mode?}
    
    GroupGen -->|Yes| RepeatPrompt[Repeat Prompt N times<br/>N = num_generations]
    GroupGen -->|No| SinglePrompt[Single Prompt]
    
    RepeatPrompt --> Rollout
    SinglePrompt --> Rollout
    
    Rollout[ROLLOUT PHASE<br/>Generate Videos] --> RewardCalc[REWARD CALCULATION<br/>Evaluate Videos]
    RewardCalc --> AdvCalc[ADVANTAGE COMPUTATION<br/>Normalize per Group]
    AdvCalc --> BestOfN[BEST-OF-N SELECTION<br/>Keep top-k and bottom-k]
    BestOfN --> PolicyUpdate[POLICY UPDATE<br/>Train with PPO-style Loss]
    PolicyUpdate --> SaveCkpt{Save Checkpoint?}
    
    SaveCkpt -->|Yes| Save[Save Model Weights]
    SaveCkpt -->|No| NextStep
    Save --> NextStep[Next Training Step]
    NextStep --> LoadData
```

## Detailed Rollout Phase

```mermaid
flowchart TD
    StartRollout([Start Rollout]) --> InitNoise[Initialize Noise z_T<br/>Random Gaussian N~0,I]
    InitNoise --> SetSchedule[Set Sigma Schedule<br/>σ: 1 → 0 in T steps]
    
    SetSchedule --> LoopStart{For each timestep t}
    
    LoopStart --> GetSigma[Get σ_t from schedule]
    GetSigma --> Forward[Transformer FORWARD PASS<br/>model.eval mode<br/>pred = Transformer z_t, text, σ_t]
    
    Forward --> FluxStep[FLUX STEP Function]
    
    FluxStep --> CalcMean[Calculate mean:<br/>μ_t = z_t + dσ · pred]
    CalcMean --> AddNoise[Add stochastic noise:<br/>z_t-1 = μ_t + ε·√Δt<br/>ε ~ N0,1]
    
    AddNoise --> LogProb[Calculate log probability:<br/>log π = -z - μ² / 2σ²<br/>- log σ - log√2π]
    
    LogProb --> Store[Store z_t-1 and log π_t]
    Store --> CheckDone{t = 0?}
    
    CheckDone -->|No| LoopStart
    CheckDone -->|Yes| Decode[VAE Decode<br/>video = VAEdecode z_0]
    
    Decode --> SaveVideo[Save Video to Disk<br/>./videos/hunyuan_rank_idx.mp4]
    SaveVideo --> ReturnResults[Return:<br/>- all_latents<br/>- all_log_probs<br/>- video]
    ReturnResults --> EndRollout([End Rollout])
    
    style FluxStep fill:#e1f5ff
    style LogProb fill:#ffe1e1
    style Store fill:#e1ffe1
```

## Reward Calculation (VideoAlign)

```mermaid
flowchart LR
    Video[Generated Video] --> VideoAlign{VideoAlign<br/>Reward Model}
    Caption[Text Caption] --> VideoAlign
    
    VideoAlign --> VQ[VQ Score<br/>Video Quality<br/>Aesthetic, Fidelity<br/>Text Alignment]
    VideoAlign --> MQ[MQ Score<br/>Motion Quality<br/>Smoothness<br/>Temporal Coherence]
    
    VQ --> Gather[Gather all rewards<br/>across GPUs]
    MQ --> Gather
    Gather --> Output[Rewards tensor<br/>shape: batch_size]
    
    style VQ fill:#ffcccc
    style MQ fill:#ccccff
```

## Advantage Computation (Group Normalization)

```mermaid
flowchart TD
    StartAdv([Start Advantage]) --> GetRewards[Get VQ and MQ rewards<br/>shape: N·G<br/>N=num_prompts<br/>G=num_generations]
    
    GetRewards --> LoopPrompts{For each prompt i}
    
    LoopPrompts --> ExtractGroup[Extract group rewards:<br/>rewards_i = rewardsi·G : i+1·G]
    ExtractGroup --> CalcStats[Calculate group statistics:<br/>μ_i = mean rewards_i<br/>σ_i = stdrewards_i + 1e-8]
    
    CalcStats --> Normalize[Normalize advantages:<br/>A_i = rewards_i - μ_i / σ_i]
    
    Normalize --> CheckNext{More prompts?}
    CheckNext -->|Yes| LoopPrompts
    CheckNext -->|No| CombineRewards[Combine VQ and MQ:<br/>total_score = α·A_VQ + β·A_MQ]
    
    CombineRewards --> EndAdv([Return Advantages])
    
    style Normalize fill:#ffffcc
    style CombineRewards fill:#ccffcc
```

## Best-of-N Selection

```mermaid
flowchart TD
    StartBest([Start Selection]) --> GetScores[Get total_score for<br/>all N×G samples]
    GetScores --> Sort[Sort by score:<br/>indices = argsort total_score]
    
    Sort --> SelectTop[Select top k samples:<br/>top_idx = indices-k:]
    Sort --> SelectBottom[Select bottom k samples:<br/>bottom_idx = indices:k]
    
    SelectTop --> Concat[Concatenate:<br/>selected = toptop, bottom]
    SelectBottom --> Concat
    
    Concat --> Shuffle[Shuffle selected indices<br/>Random permutation]
    Shuffle --> Filter[Filter all data:<br/>latents = latentsselected<br/>log_probs = log_probsselected<br/>advantages = advantagesselected]
    
    Filter --> EndBest([Return Filtered Data])
    
    style SelectTop fill:#90EE90
    style SelectBottom fill:#FFB6C6
```

## Policy Update (PPO-style Training)

```mermaid
flowchart TD
    StartUpdate([Start Update]) --> Shuffle[Randomly shuffle timesteps<br/>for each sample]
    Shuffle --> SelectSteps[Select fraction of timesteps:<br/>num_train = T × timestep_fraction]
    
    SelectSteps --> LoopSamples{For each sample}
    LoopSamples --> LoopTime{For each selected<br/>timestep t}
    
    LoopTime --> Recompute[RECOMPUTE log prob<br/>model.train mode<br/>new_log_π = grpo_one_step...]
    
    Recompute --> CalcRatio[Calculate importance ratio:<br/>ratio = expnew_log_π - old_log_π]
    
    CalcRatio --> ClipAdv[Clip advantages:<br/>A = clampadv, -5, 5]
    
    ClipAdv --> CalcLossVQ[VQ Loss:<br/>L1 = -A_VQ · ratio<br/>L2 = -A_VQ · clamp ratio, 1-ε, 1+ε<br/>L_VQ = meanmax L1, L2]
    
    CalcLossVQ --> CalcLossMQ[MQ Loss:<br/>Similar clipped loss for MQ]
    
    CalcLossMQ --> CombineLoss[Combined Loss:<br/>L = α·L_VQ + β·L_MQ]
    CombineLoss --> Scale[Scale by:<br/>L = L / grad_accum × num_train]
    
    Scale --> Backward[L.backward<br/>Accumulate gradients]
    
    Backward --> CheckAccum{Accumulated<br/>enough?}
    CheckAccum -->|No| LoopTime
    CheckAccum -->|Yes| ClipGrad[Clip gradients:<br/>grad_norm = clip_gradmax_norm]
    
    ClipGrad --> OptStep[optimizer.step<br/>lr_scheduler.step<br/>optimizer.zero_grad]
    
    OptStep --> CheckMore{More samples?}
    CheckMore -->|Yes| LoopSamples
    CheckMore -->|No| EndUpdate([End Update])
    
    style CalcRatio fill:#FFE4B5
    style CombineLoss fill:#E6E6FA
    style Backward fill:#98FB98
```

## Complete GRPO Pipeline (High-Level)

```mermaid
graph TB
    subgraph Input
        P[Prompt]
        T[Text Embeddings]
    end
    
    subgraph "Phase 1: Rollout (Inference Mode)"
        G1[Generate Video 1]
        G2[Generate Video 2]
        GN[Generate Video N]
        L1[Log π₁]
        L2[Log π₂]
        LN[Log πₙ]
    end
    
    subgraph "Phase 2: Evaluation"
        V1[VideoAlign Video 1]
        V2[VideoAlign Video 2]
        VN[VideoAlign Video N]
        R1[VQ₁, MQ₁]
        R2[VQ₂, MQ₂]
        RN[VQₙ, MQₙ]
    end
    
    subgraph "Phase 3: Advantage"
        Mean[μ = mean rewards]
        Std[σ = std rewards]
        A1[Advantage₁ = r₁-μ/σ]
        A2[Advantage₂ = r₂-μ/σ]
        AN[Advantageₙ = rₙ-μ/σ]
    end
    
    subgraph "Phase 4: Selection"
        Best[Top k samples]
        Worst[Bottom k samples]
    end
    
    subgraph "Phase 5: Training (Train Mode)"
        Rerun[Recompute log π']
        PPO[PPO Clipped Loss:<br/>L = -A × min ratio, clip ratio]
        Update[Update Model Weights]
    end
    
    P --> G1
    P --> G2
    P --> GN
    T --> G1
    T --> G2
    T --> GN
    
    G1 --> L1
    G2 --> L2
    GN --> LN
    
    G1 --> V1
    G2 --> V2
    GN --> VN
    
    V1 --> R1
    V2 --> R2
    VN --> RN
    
    R1 --> Mean
    R2 --> Mean
    RN --> Mean
    R1 --> Std
    R2 --> Std
    RN --> Std
    
    Mean --> A1
    Mean --> A2
    Mean --> AN
    Std --> A1
    Std --> A2
    Std --> AN
    
    A1 --> Best
    A2 --> Worst
    AN --> Best
    
    Best --> Rerun
    Worst --> Rerun
    L1 --> Rerun
    L2 --> Rerun
    LN --> Rerun
    
    Rerun --> PPO
    A1 --> PPO
    A2 --> PPO
    
    PPO --> Update
    
    style G1 fill:#e1f5ff
    style G2 fill:#e1f5ff
    style GN fill:#e1f5ff
    style V1 fill:#ffe1e1
    style V2 fill:#ffe1e1
    style VN fill:#ffe1e1
    style A1 fill:#ffffcc
    style A2 fill:#ffffcc
    style AN fill:#ffffcc
    style PPO fill:#98FB98
    style Update fill:#FFB6C1
```

## Key Equations

### Log Probability (Gaussian)
```
log π(z_{t-1} | z_t) = -||z_{t-1} - μ_t||² / (2σ_t²) - log(σ_t) - log(√(2π))

where:
  μ_t = z_t + dσ · model_output
  σ_t = η · √(Δt)
```

### Advantage (Group Normalization)
```
A_i = (r_i - μ_group) / (σ_group + ε)

where:
  μ_group = mean of rewards for same prompt
  σ_group = std of rewards for same prompt
```

### PPO Clipped Loss
```
L = mean(max(-A · ratio, -A · clip(ratio, 1-ε, 1+ε)))

where:
  ratio = exp(log π_new - log π_old)
  ε = clip_range (typically 1e-4)
```

### Combined Loss (Multi-Reward)
```
L_total = α · L_VQ + β · L_MQ

where:
  α = vq_coef (video quality coefficient)
  β = mq_coef (motion quality coefficient)
```

