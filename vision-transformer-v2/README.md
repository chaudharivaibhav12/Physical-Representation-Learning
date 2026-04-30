# Vision Transformer V2 — VICReg on Active Matter

Improved version of Vision Transformer V1. Same 3D ViT encoder + VICReg objective, with significant infrastructure and training improvements for longer, preemption-safe training on spot GPU instances.

---

## What Changed from V1

### 1. Training Scale
| Property | V1 | V2 |
|---|---|---|
| Epochs | 20 | **100** |
| Batch size (per GPU) | 4 | **8** |
| Effective batch size | 32 | **64** |
| stride (train samples) | 4 → 2,275 clips | **1 → 8,750 clips** |

V2 trains on the full dataset at stride=1, matching the project spec of 8,750 training samples. V1 used stride=4 which only saw ~26% of available training data.

### 2. VICReg Loss Weights
| Term | V1 | V2 |
|---|---|---|
| Invariance | 25.0 | 25.0 |
| Variance | **50.0** | **25.0** |
| Covariance | 1.0 | 1.0 |

V2 reduces variance weight from 50 → 25 (matching the original VICReg paper defaults) since the larger dataset and batch size provide a stronger collapse-prevention signal naturally.

### 3. Preemption-Safe Checkpointing
V1 only saved epoch-level checkpoints, losing up to one full epoch of training on spot instance preemption. V2 adds:
- **Step-level checkpoints** every 20 steps — at most 20 steps lost on preemption
- **global_step saved/restored** in checkpoint — training continues from exact step, not just epoch
- **SIGUSR1 signal handler** — SLURM sends SIGUSR1 90s before preemption; Python catches it, saves checkpoint, then bash requeues the job
- **Instant batch-skip on resume** — uses a deterministic per-epoch `torch.Generator` seed + `Subset` to skip already-processed batches in O(1), no data loading wasted

### 4. Deterministic Epoch Shuffling
V1 used a standard `shuffle=True` DataLoader — non-reproducible across restarts. V2 seeds each epoch's shuffle with `epoch` via `torch.Generator`, so resumed runs see identical data order:

```python
g = torch.Generator()
g.manual_seed(epoch)
indices = torch.randperm(len(dataset), generator=g).tolist()
subset  = Subset(dataset, indices[skip_batches * batch_size:])
```

### 5. W&B Run Persistence
V1 created a new W&B run on every restart, fragmenting training logs. V2 saves the W&B run ID to disk and resumes it across preemptions — all metrics appear on a single continuous curve.

### 6. Evaluation Fix (stride=1)
V1 evaluate.py used default stride (stride=4 → 2,275 train embeddings). V2 explicitly passes `stride=1` → 8,750/1,200/1,300 samples matching the project spec.

---

## Architecture (unchanged from V1)

```
Training:
  view1 (B, 11, 16, 224, 224)
      → PatchEmbed3D (Conv3D, 32×32 patch, tubelet=2)
      → 392 tokens (B, 392, 384)
      → 6× TransformerBlock (pre-norm, MHA + MLP)
      → LayerNorm → mean pool
      → (B, 384)
      → ProjectionMLP (384 → 2048 → 2048 → 2048)
      → z1 (B, 2048)
                                ┐
  view2 (same clip, different   ├─→ VICReg Loss
  augmentation) → same path     │
      → z2 (B, 2048)           ┘

Evaluation (projector discarded):
  clip → Encoder → mean pool → (B, 384) → Linear Probe / kNN
```

### Encoder
| Property | Value |
|---|---|
| Embed dim | 384 |
| Depth | 6 transformer blocks |
| Attention heads | 6 (head dim = 64) |
| MLP ratio | 4× (hidden = 1536) |
| Positional embedding | Fixed 3D sinusoidal |
| Token grid | 8T × 7H × 7W = 392 tokens |
| Output | mean-pool → (B, 384) |

### Projector (training only, discarded at eval)
```
384 → Linear(2048) → BN → ReLU → Linear(2048) → BN → ReLU → Linear(2048)
```

### VICReg Loss
| Term | Weight |
|---|---|
| Invariance | 25.0 |
| Variance | 25.0 |
| Covariance | 1.0 |

---

## Parameter Count

| Component | Parameters |
|---|---|
| Encoder | ~11M |
| Projector | ~12.6M |
| **Total trainable** | **~23.6M** |

---

## Dataset: active_matter

| Property | Value |
|---|---|
| Source | HuggingFace `polymathic-ai/active_matter` |
| Input channels | 11 (concentration, velocity, orientation tensor, strain-rate tensor) |
| Spatial resolution | 224×224 (center-cropped from 256×256) |
| Temporal length | 16 frames per clip |
| Train samples | **8,750** (stride=1) |
| Validation samples | 1,200 |
| Test samples | 1,300 |
| Physical parameters | α (active dipole strength, 5 values), ζ (steric alignment, 9 values) |

### Augmentation (training only)
- Random crop 224×224 from 256×256
- Random horizontal and vertical flip
- Random 90° rotation
- Gaussian noise (std=1.0)

### Normalization
Per-sample, per-channel z-score normalization across T×H×W.

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 100 |
| Batch size (per GPU) | 8 |
| Effective batch size | 64 (8 gradient accumulation steps) |
| Learning rate | 1e-3 |
| LR schedule | Cosine annealing with 5-epoch warmup |
| Weight decay | 0.05 |
| Gradient clip | 1.0 |
| Optimizer | AdamW |
| Mixed precision | bfloat16 |
| Step checkpoint | every 20 steps |
| Epoch checkpoint | every 5 epochs |

---

## Evaluation Results

Frozen encoder (384-dim), z-score normalized targets, MSE.

### Validation MSE
| Method | α | ζ |
|---|---|---|
| Linear Probe | 0.0613 | 0.3586 |
| kNN (k=20) | 0.0379 | 0.2754 |

### Test MSE
| Method | α | ζ |
|---|---|---|
| Linear Probe | 0.0696 | 0.3412 |
| kNN (k=20) | 0.0415 | 0.3030 |

Random baseline ≈ 1.0. Lower is better.

---

## Files

| File | Description |
|---|---|
| `model.py` | Full model: PatchEmbed3D, ViTEncoder, ProjectionMLP, VICRegLoss, VICReg |
| `dataset.py` | ActiveMatterDataset + ActiveMatterEval — HDF5 loading, augmentation |
| `train.py` | Training loop with step checkpointing, SIGUSR1 handler, W&B persistence |
| `evaluate.py` | Linear probe + kNN regression on frozen encoder (stride=1, 8750 samples) |
| `submit.slurm` | SLURM job script with preemption requeue |
| `eval.slurm` | SLURM eval job script |

---

## Running

**Training:**
```bash
python train.py
python train.py --resume /scratch/sb10583/checkpoints/vicreg-v4/latest.pt
```

**Evaluation:**
```bash
python evaluate.py --checkpoint /scratch/sb10583/checkpoints/vicreg-v4/best.pt --test
```

**HPC:**
```bash
sbatch submit.slurm
sbatch eval.slurm
```
