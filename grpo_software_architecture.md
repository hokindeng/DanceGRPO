# GRPO Software Architecture - Engineering Perspective

## System Architecture Overview

```mermaid
graph TB
    subgraph "Entry Point"
        Main[train_grpo_hunyuan.py<br/>main function]
    end
    
    subgraph "Distributed Training Setup"
        DistInit[torch.distributed<br/>init_process_group NCCL]
        SeqPar[Sequence Parallel<br/>initialize_sequence_parallel_state]
        FSDP[FSDP Wrapper<br/>FullyShardedDataParallel]
    end
    
    subgraph "Model Components"
        Transformer[Transformer Model<br/>HunyuanTransformer3D<br/>or FluxTransformer2D]
        VAE[VAE Decoder<br/>AutoencoderKL]
        RewardModel[Reward Model<br/>VideoVLMRewardInference<br/>or HPSv2]
    end
    
    subgraph "Data Pipeline"
        Dataset[LatentDataset<br/>Loads preprocessed<br/>text embeddings]
        Sampler[DistributedSampler<br/>Shard data across GPUs]
        Loader[DataLoader<br/>Parallel workers]
        Wrapper[sp_parallel_dataloader_wrapper<br/>Sequence parallel batching]
    end
    
    subgraph "Training Loop"
        Rollout[sample_reference_model<br/>Generate videos + log probs]
        Reward[Calculate rewards<br/>VideoAlign/HPS inference]
        Advantage[Compute advantages<br/>Group normalization]
        Filter[Best-of-N selection<br/>Filter samples]
        Update[train_one_step<br/>PPO loss + backprop]
    end
    
    subgraph "Optimization"
        Optimizer[AdamW Optimizer<br/>FSDP-aware]
        Scheduler[LR Scheduler<br/>Warmup + decay]
        GradClip[Gradient Clipping<br/>clip_grad_norm_]
    end
    
    subgraph "I/O & Monitoring"
        Checkpoint[save_checkpoint<br/>FSDP state dict]
        WandB[Weights & Biases<br/>Logging]
        VideoIO[Video I/O<br/>export_to_video]
    end
    
    Main --> DistInit
    Main --> SeqPar
    DistInit --> FSDP
    SeqPar --> FSDP
    
    FSDP --> Transformer
    Main --> VAE
    Main --> RewardModel
    
    Main --> Dataset
    Dataset --> Sampler
    Sampler --> Loader
    Loader --> Wrapper
    
    Wrapper --> Rollout
    Rollout --> Transformer
    Rollout --> VAE
    Rollout --> VideoIO
    VideoIO --> Reward
    Reward --> RewardModel
    Reward --> Advantage
    Advantage --> Filter
    Filter --> Update
    Update --> Transformer
    Update --> Optimizer
    Optimizer --> Scheduler
    Update --> GradClip
    
    Update --> Checkpoint
    Update --> WandB
    
    style Transformer fill:#FFE4B5
    style FSDP fill:#98FB98
    style Rollout fill:#E6E6FA
    style Update fill:#FFB6C1
```

## Class and Module Structure

```mermaid
classDiagram
    class TrainScript {
        +main(args)
        +train_one_step()
        +sample_reference_model()
        +flux_step()
        +grpo_one_step()
        +run_sample_step()
    }
    
    class LatentDataset {
        -data_json_path: str
        -num_frames: int
        -cfg: float
        +__getitem__(idx)
        +__len__()
    }
    
    class TransformerModel {
        <<FSDP Wrapped>>
        +forward(hidden_states, encoder_hidden_states, timestep)
        +train()
        +eval()
        +clip_grad_norm_(max_norm)
    }
    
    class VAEDecoder {
        +decode(latents)
        +enable_tiling()
    }
    
    class RewardModel {
        +reward(videos, captions)
        +inference_mode()
    }
    
    class FSDPUtils {
        +get_dit_fsdp_kwargs()
        +apply_fsdp_checkpointing()
        +get_model_state_dict()
        +set_model_state_dict()
    }
    
    class CheckpointManager {
        +save_checkpoint(model, rank, dir, step, epoch)
        +save_lora_checkpoint()
        +resume_from_checkpoint()
    }
    
    class ParallelStates {
        +initialize_sequence_parallel_state(sp_size)
        +get_sequence_parallel_state()
        +destroy_sequence_parallel_group()
    }
    
    class Communications {
        +sp_parallel_dataloader_wrapper()
        +broadcast()
        +gather_tensor()
    }
    
    class FSDP_EMA {
        -decay: float
        -ema_state_dict_rank0: dict
        +__init__(model, decay, rank)
        +update(model)
        +use_ema_weights(model)
    }
    
    TrainScript --> LatentDataset
    TrainScript --> TransformerModel
    TrainScript --> VAEDecoder
    TrainScript --> RewardModel
    TrainScript --> FSDPUtils
    TrainScript --> CheckpointManager
    TrainScript --> ParallelStates
    TrainScript --> Communications
    TrainScript --> FSDP_EMA
    
    TransformerModel --> FSDPUtils
    CheckpointManager --> FSDPUtils
```

## Data Structure Flow

```mermaid
flowchart LR
    subgraph "Input Data Structures"
        JSON[data.json<br/>List of dicts<br/>path, caption]
        Embed[Preprocessed Embeddings<br/>.pt files<br/>encoder_hidden_states<br/>attention_mask]
    end
    
    subgraph "Batch Structure"
        Batch[Batch Dict:<br/>encoder_hidden_states: BxLxD<br/>encoder_attention_mask: BxL<br/>caption: List of strings]
    end
    
    subgraph "Rollout Outputs"
        Latents[all_latents:<br/>Tensor Bx T+1 xCxHxW<br/>All intermediate states]
        LogProbs[all_log_probs:<br/>Tensor BxT<br/>Log π at each step]
        Videos[videos:<br/>List of arrays<br/>Decoded videos]
        Rewards[rewards:<br/>Tensor B<br/>VQ and MQ scores]
    end
    
    subgraph "Training Tensors"
        Samples[samples dict:<br/>timesteps: BxT<br/>latents: BxTxCxHxW<br/>next_latents: BxTxCxHxW<br/>log_probs: BxT<br/>vq_advantages: B<br/>mq_advantages: B<br/>encoder_hidden_states: BxLxD<br/>encoder_attention_mask: BxL]
    end
    
    subgraph "Filtered Tensors"
        Selected[selected_samples:<br/>Same structure<br/>but batch_size = bestofn<br/>Contains top-k and bottom-k]
    end
    
    JSON --> Embed
    Embed --> Batch
    Batch --> Latents
    Batch --> LogProbs
    Latents --> Videos
    Videos --> Rewards
    
    Latents --> Samples
    LogProbs --> Samples
    Rewards --> Samples
    Batch --> Samples
    
    Samples --> Selected
    Selected --> Loss[Loss Computation<br/>Scalar tensor]
```

## Memory Layout and Optimization

```mermaid
graph TB
    subgraph "GPU Memory Layout (per GPU)"
        subgraph "Model Shards FSDP"
            Shard1[Transformer Shard 1 of N<br/>Parameters + Gradients<br/>about 10-30 GB]
            Shard2[Transformer Shard 2 of N]
            ShardN[Transformer Shard N of N]
        end
        
        subgraph "Activation Memory"
            Forward[Forward Activations<br/>Latents: BxTxCxHxW<br/>about 5-20 GB]
            Backward[Backward Gradients<br/>about 5-20 GB]
        end
        
        subgraph "Optimizer States"
            Adam1[AdamW State Shard<br/>First Moment<br/>about 10-30 GB]
            Adam2[AdamW State Shard<br/>Second Moment<br/>about 10-30 GB]
        end
        
        subgraph "Temporary Buffers"
            AllGather[All-Gather Buffer<br/>Full model params<br/>about 2-5 GB temporary]
            VideoBuffer[Decoded Videos<br/>about 2-10 GB temporary]
        end
    end
    
    subgraph "CPU Memory"
        EMA[EMA Weights Rank 0<br/>Full model copy<br/>about 20-50 GB]
        Checkpoint[Checkpoint Buffer<br/>State dict<br/>about 20-50 GB]
    end
    
    subgraph "Disk I/O"
        Videos[Videos on Disk<br/>./videos/video.mp4<br/>about 1-5 GB]
        Ckpts[Checkpoints<br/>./output_dir/checkpoints<br/>about 20-50 GB per ckpt]
        Logs[Logs<br/>reward.txt, vq_reward.txt]
    end
    
    Shard1 -.->|"Gradient Sync"| Shard2
    Shard2 -.->|"Gradient Sync"| ShardN
    
    Forward -->|"Accumulate"| Backward
    Backward -->|"Update"| Adam1
    Backward -->|"Update"| Adam2
    
    Adam1 -->|"Step"| Shard1
    Adam2 -->|"Step"| Shard1
    
    AllGather -.->|"Temporary"| Shard1
    VideoBuffer -.->|"Decode"| Forward
    
    Shard1 -.->|"Copy Rank 0"| EMA
    EMA -.->|"Periodic"| Checkpoint
    Checkpoint -->|"Write"| Ckpts
    
    Forward -.->|"Decode & Save"| Videos
    
    style Shard1 fill:#FFE4B5
    style Forward fill:#E6E6FA
    style Adam1 fill:#98FB98
    style EMA fill:#FFB6C1
```

## Thread and Process Model

```mermaid
graph TB
    subgraph "Multi-Node Cluster"
        Node1[Node 1<br/>8 GPUs]
        Node2[Node 2<br/>8 GPUs]
        NodeN[Node N<br/>8 GPUs]
    end
    
    Node1 -.->|"NCCL All-Reduce<br/>Gradients"| Node2
    Node2 -.->|"NCCL All-Reduce"| NodeN
    
    subgraph "Node 1 Detail"
        subgraph "Rank 0 GPU 0"
            MP0[Main Process<br/>torch.distributed]
            DW0[DataLoader Workers<br/>num_workers threads]
            MP0 --> DW0
        end
        
        subgraph "Rank 1 GPU 1"
            MP1[Main Process]
            DW1[DataLoader Workers]
            MP1 --> DW1
        end
        
        subgraph "Rank 7 GPU 7"
            MP7[Main Process]
            DW7[DataLoader Workers]
            MP7 --> DW7
        end
    end
    
    subgraph "Sequence Parallel Groups"
        SP1[SP Group 1<br/>Ranks 0 1 2 3]
        SP2[SP Group 2<br/>Ranks 4 5 6 7]
    end
    
    MP0 -.->|"Sequence Parallel"| MP1
    MP1 -.->|"Sequence Parallel"| MP7
    
    MP0 --> SP1
    MP1 --> SP1
    MP7 --> SP2
    
    subgraph "Synchronization Points"
        Barrier1[dist.barrier<br/>Before checkpoint]
        Barrier2[dist.barrier<br/>After training step]
        AllReduce[dist.all_reduce<br/>Loss aggregation]
        AllGather[dist.all_gather<br/>Reward gathering]
    end
    
    MP0 -.-> Barrier1
    MP1 -.-> Barrier1
    MP7 -.-> Barrier1
    
    MP0 -.-> AllReduce
    MP1 -.-> AllReduce
    
    style MP0 fill:#FFE4B5
    style SP1 fill:#E6E6FA
    style Barrier1 fill:#98FB98
```

## Training Loop State Machine

```mermaid
stateDiagram-v2
    [*] --> Initialize
    
    Initialize --> LoadCheckpoint: resume_from_checkpoint?
    Initialize --> FreshStart: no checkpoint
    LoadCheckpoint --> DataLoading
    FreshStart --> DataLoading
    
    DataLoading --> GroupExpansion: use group True
    DataLoading --> RolloutPhase: use group False
    GroupExpansion --> RolloutPhase
    
    RolloutPhase --> NoiseInit: for each sample
    NoiseInit --> DenoisingLoop
    
    DenoisingLoop --> ForwardPass: t = T
    ForwardPass --> FluxStep
    FluxStep --> LogProbCalc
    LogProbCalc --> StoreState
    StoreState --> DenoisingLoop: t > 0
    StoreState --> VAEDecode: t = 0
    
    VAEDecode --> SaveVideo
    SaveVideo --> RewardInference
    RewardInference --> CheckAllSamples: more samples?
    CheckAllSamples --> NoiseInit: yes
    CheckAllSamples --> AdvantageCalc: no
    
    AdvantageCalc --> BestOfNFilter
    BestOfNFilter --> TimestepShuffle
    
    TimestepShuffle --> PolicyTraining
    PolicyTraining --> RecomputeLogProb: for each timestep
    RecomputeLogProb --> RatioCalc
    RatioCalc --> LossCalc
    LossCalc --> Backward
    Backward --> GradAccum: accumulate more?
    GradAccum --> RecomputeLogProb: yes
    GradAccum --> OptimizerStep: no
    
    OptimizerStep --> EMAUpdate: use ema True
    OptimizerStep --> CheckSavepoint: use ema False
    EMAUpdate --> CheckSavepoint
    
    CheckSavepoint --> SaveCheckpoint: step mod save_steps equals 0
    CheckSavepoint --> LogMetrics: otherwise
    SaveCheckpoint --> LogMetrics
    
    LogMetrics --> CheckDone: step < max_steps?
    CheckDone --> DataLoading: continue
    CheckDone --> Finalize: done
    
    Finalize --> [*]
    
    note right of RolloutPhase
        Model in eval mode
        torch no_grad
    end note
    
    note right of PolicyTraining
        Model in train mode
        Gradients enabled
    end note
    
    note right of OptimizerStep
        Gradient clipping
        LR scheduling
    end note
```

## File System Organization

```mermaid
graph TB
    subgraph "Project Root"
        Config[config files<br/>args, hyperparams]
        Scripts[scripts/<br/>Shell scripts for training]
    end
    
    subgraph "Source Code"
        Train[fastvideo/train_grpo_*.py<br/>Main training scripts]
        Dataset[fastvideo/dataset/<br/>Data loading code]
        Models[fastvideo/models/<br/>Model definitions]
        Utils[fastvideo/utils/<br/>Helper functions]
    end
    
    subgraph "Data Directory"
        PretrainedModels[data/<br/>Pretrained model weights]
        Embeddings[preprocessed/<br/>Text embeddings .pt files]
        DataJSON[data.json<br/>Dataset metadata]
    end
    
    subgraph "Runtime Outputs"
        Videos[videos/<br/>Generated videos per rank<br/>hunyuan_rank_idx.mp4]
        Checkpoints[output_dir/<br/>checkpoint-step-epoch/]
        Logs[Logs<br/>reward.txt, vq_reward.txt, mq_reward.txt]
    end
    
    subgraph "Checkpoint Structure"
        ModelState[model state dict<br/>FSDP format]
        OptimizerState[optimizer state<br/>AdamW states]
        SchedulerState[lr_scheduler state]
        EMAState[EMA state dict<br/>Rank 0 only]
    end
    
    Config --> Train
    Scripts --> Train
    Train --> Dataset
    Train --> Models
    Train --> Utils
    
    DataJSON --> Dataset
    Embeddings --> Dataset
    PretrainedModels --> Models
    
    Train --> Videos
    Train --> Checkpoints
    Train --> Logs
    
    Checkpoints --> ModelState
    Checkpoints --> OptimizerState
    Checkpoints --> SchedulerState
    Checkpoints --> EMAState
```

## Error Handling and Robustness

```mermaid
flowchart TD
    Start[Training Step] --> TryBlock{Try Block}
    
    TryBlock --> Rollout[Rollout Generation]
    Rollout --> RewardCalc[Reward Calculation]
    
    RewardCalc --> CatchReward{Reward<br/>Exception?}
    CatchReward -->|Yes| DefaultReward[Set reward = -1.0<br/>Log error<br/>Continue training]
    CatchReward -->|No| AdvCalc[Advantage Calculation]
    
    DefaultReward --> AdvCalc
    
    AdvCalc --> Training[Policy Update]
    Training --> CatchNaN{NaN in<br/>gradients?}
    
    CatchNaN -->|Yes| SkipStep[Skip optimizer step<br/>Log warning<br/>Continue to next batch]
    CatchNaN -->|No| OptStep[Optimizer Step]
    
    SkipStep --> Barrier
    OptStep --> Barrier
    
    Barrier[dist.barrier<br/>Sync all ranks] --> CheckMemory{GPU OOM?}
    
    CheckMemory -->|Yes| Restart[Process crash<br/>Auto-restart from<br/>last checkpoint]
    CheckMemory -->|No| SaveCheck{Time to<br/>checkpoint?}
    
    SaveCheck -->|Yes| TrySave{Try Save}
    SaveCheck -->|No| NextStep
    
    TrySave --> CatchIO{I/O Error?}
    CatchIO -->|Yes| RetryIO[Retry 3 times<br/>with exponential backoff]
    CatchIO -->|No| NextStep
    
    RetryIO --> FinalFail{All retries<br/>failed?}
    FinalFail -->|Yes| LogError[Log error but continue<br/>Skip this checkpoint]
    FinalFail -->|No| NextStep
    
    LogError --> NextStep
    NextStep[Next Training Step] --> End
    Restart --> LoadCheckpoint[Load last checkpoint<br/>Resume training]
    LoadCheckpoint --> End
    
    style DefaultReward fill:#FFE4B5
    style SkipStep fill:#FFB6C1
    style Restart fill:#FF6B6B
    style RetryIO fill:#FFA07A
```

## Performance Optimization Patterns

```mermaid
graph TB
    subgraph "Memory Optimizations"
        GradCheckpoint[Gradient Checkpointing<br/>Trade compute for memory<br/>selective_checkpointing ratio]
        CPUOffload[CPU Offload<br/>use_cpu_offload flag<br/>Offload optimizer states]
        VAETiling[VAE Tiling<br/>vae.enable_tiling<br/>Decode in tiles]
        MixedPrecision[Mixed Precision<br/>torch.autocastbf16<br/>Reduce memory by 50%]
    end
    
    subgraph "Compute Optimizations"
        TF32[TF32 Matmul<br/>torch.backends.cuda.matmul.allow_tf32<br/>~8x speedup]
        FlashAttn[Flash Attention<br/>Memory-efficient attention<br/>O n instead of O n²]
        GradAccum[Gradient Accumulation<br/>Simulate larger batch size<br/>without OOM]
        TimestepFraction[Timestep Sampling<br/>timestep_fraction<br/>Train on subset of timesteps]
    end
    
    subgraph "I/O Optimizations"
        NumWorkers[DataLoader Workers<br/>dataloader_num_workers<br/>Parallel data loading]
        PinMemory[Pin Memory<br/>pin_memory=True<br/>Faster CPU→GPU transfer]
        Prefetch[Prefetch<br/>Preload next batch<br/>while training]
        AsyncIO[Async Video I/O<br/>Save videos in background]
    end
    
    subgraph "Distributed Optimizations"
        FSDP[FSDP Sharding<br/>Shard model + optimizer<br/>across GPUs]
        SeqParallel[Sequence Parallel<br/>Split temporal dimension<br/>across GPUs]
        NCCL[NCCL Communication<br/>Optimized GPU-GPU<br/>All-Reduce/All-Gather]
        OverlapComm[Overlap Compute & Comm<br/>Backward pass overlaps<br/>gradient sync]
    end
    
    subgraph "Algorithmic Optimizations"
        BestOfN[Best-of-N Filtering<br/>Train on top-k + bottom-k<br/>instead of all samples]
        GroupNorm[Group Normalization<br/>Per-prompt advantage<br/>Better learning signal]
        EarlyStopping[Early Stopping<br/>Stop if reward plateaus]
        SameNoise[Same Noise Init<br/>init_same_noise<br/>Reduce variance]
    end
    
    GradCheckpoint --> Memory[Reduce Peak Memory<br/>Enable larger models]
    CPUOffload --> Memory
    VAETiling --> Memory
    MixedPrecision --> Memory
    
    TF32 --> Speed[Increase Throughput<br/>More steps/sec]
    FlashAttn --> Speed
    GradAccum --> Speed
    TimestepFraction --> Speed
    
    NumWorkers --> Pipeline[Reduce Data Bottleneck]
    PinMemory --> Pipeline
    Prefetch --> Pipeline
    AsyncIO --> Pipeline
    
    FSDP --> Scale[Scale to Large Models<br/>Scale to Many GPUs]
    SeqParallel --> Scale
    NCCL --> Scale
    OverlapComm --> Scale
    
    BestOfN --> Sample[Improve Sample Efficiency<br/>Better reward signal]
    GroupNorm --> Sample
    EarlyStopping --> Sample
    SameNoise --> Sample
    
    style Memory fill:#FFE4B5
    style Speed fill:#98FB98
    style Pipeline fill:#E6E6FA
    style Scale fill:#FFB6C1
    style Sample fill:#FFA07A
```

## Key Configuration Parameters

```yaml
# Distributed Training
world_size: 16                      # Total GPUs
sp_size: 2                          # Sequence parallel group size
fsdp_sharding_strategy: "full"      # FSDP sharding mode

# Memory Management
use_cpu_offload: false              # Offload optimizer to CPU
gradient_checkpointing: true        # Enable gradient checkpointing
selective_checkpointing: 0.5        # Checkpoint 50% of layers
master_weight_type: "fp32"          # Master weights precision

# Training Hyperparameters
learning_rate: 1e-5                 # AdamW learning rate
weight_decay: 0.01                  # L2 regularization
max_grad_norm: 2.0                  # Gradient clipping threshold
gradient_accumulation_steps: 4      # Accumulate before update

# GRPO Specific
num_generations: 16                 # Videos per prompt
bestofn: 8                          # Keep top+bottom k
use_group: true                     # Group normalization
clip_range: 1e-4                    # PPO clipping epsilon
adv_clip_max: 5.0                   # Advantage clipping
timestep_fraction: 1.0              # Fraction of timesteps to train

# Video Generation
sampling_steps: 50                  # Denoising steps
h: 720                              # Video height
w: 1280                             # Video width
t: 49                               # Number of frames
fps: 24                             # Frames per second
eta: 1.0                            # SDE noise scale

# Reward Model
vq_coef: 1.0                        # Video quality coefficient
mq_coef: 1.0                        # Motion quality coefficient
use_videoalign: true                # Use VideoAlign reward

# I/O
dataloader_num_workers: 10          # DataLoader workers
checkpointing_steps: 20             # Save every N steps
max_train_steps: 200                # Total training steps

# EMA
use_ema: true                       # Enable EMA
ema_decay: 0.995                    # EMA decay rate
```

## Common Pitfalls and Solutions

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| **OOM during rollout** | CUDA out of memory in generation phase | Reduce `num_generations`, enable `vae.enable_tiling()`, reduce `sampling_steps` |
| **OOM during training** | CUDA out of memory in backward pass | Increase `gradient_accumulation_steps`, enable `gradient_checkpointing`, use `cpu_offload` |
| **Slow data loading** | GPU idle waiting for data | Increase `dataloader_num_workers`, enable `pin_memory`, precompute embeddings |
| **Gradient explosion** | Loss becomes NaN | Reduce `learning_rate`, decrease `max_grad_norm`, check reward model outputs |
| **Reward collapse** | All rewards become similar | Use `init_same_noise=True`, increase `num_generations`, check reward model quality |
| **Slow convergence** | Reward doesn't improve | Increase `learning_rate`, reduce `clip_range`, increase batch size |
| **Checkpoint corruption** | Can't load checkpoint | Add checksum verification, retry logic with exponential backoff |
| **GPU hang** | Process freezes | Check NCCL timeout, add `dist.barrier()` debugging, verify network connectivity |
| **Memory leak** | Memory usage grows over time | Call `torch.cuda.empty_cache()`, check for circular references, profile with `torch.cuda.memory_summary()` |
| **Uneven GPU utilization** | Some GPUs idle | Balance data sharding, check sequence parallel split, verify FSDP strategy |

## Debugging Strategies

```python
# 1. Verify probability ratio
if step == 1 and local_rank == 0:
    print(f"Ratio at step 1: {ratio}")  # Should be ~1.0

# 2. Check gradient flow
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm()}")

# 3. Monitor memory
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# 4. Profile NCCL
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"

# 5. Check reward distribution
print(f"Rewards: min={rewards.min()}, max={rewards.max()}, mean={rewards.mean()}")
print(f"Advantages: min={advantages.min()}, max={advantages.max()}")

# 6. Validate data
assert not torch.isnan(latents).any(), "NaN in latents"
assert not torch.isinf(log_probs).any(), "Inf in log_probs"

# 7. Time profiling
import time
start = time.time()
# ... code block ...
print(f"Block took {time.time() - start:.2f}s")
```

