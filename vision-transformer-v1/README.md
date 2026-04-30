# Vision Transformer V1 — VICReg on Active Matter

Self-supervised representation learning on physical simulations using a 3D Vision Transformer encoder trained with the VICReg objective.

---

## Architecture Overview

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

---

## Components

### 1. PatchEmbed3D
Converts a spatiotemporal clip into tokens using a single Conv3D layer.

| Property | Value |
|---|---|
| Kernel | 2 (temporal) × 32 × 32 (spatial) |
| Stride | same as kernel (non-overlapping) |
| Token grid | 8T × 7H × 7W = **392 tokens** |
| Output dim | 384 |
| Post-norm | LayerNorm |

### 2. Positional Embedding
Fixed 3D sinusoidal positional embedding — not learned. Splits embed_dim into 3 equal parts for temporal, height, and width axes independently.

### 3. ViT Encoder
| Property | Value |
|---|---|
| Embed dim | 384 |
| Depth | 6 transformer blocks |
| Attention heads | 6 (head dim = 64) |
| MLP ratio | 4× (hidden dim = 1536) |
| Pre-norm | Yes (LayerNorm before attention and MLP) |
| Pooling | Global mean pool over all 392 tokens → (B, 384) |

### 4. Projection MLP (training only)
Expands the encoder output to high-dimensional space for VICReg loss. Discarded after training.

```
384 → Linear(2048) → BN → ReLU
    → Linear(2048) → BN → ReLU
    → Linear(2048)
→ z (B, 2048)
```

### 5. VICReg Loss
Applied between the two projected views z1 and z2.

| Term | Weight | Purpose |
|---|---|---|
| Invariance | 25.0 | MSE(z1, z2) — pull same-clip views together |
| Variance | 50.0 | Keep per-dim std ≥ 1 — prevent collapse |
| Covariance | 1.0 | Decorrelate embedding dimensions |

---

## Parameter Count

| Component | Parameters |
|---|---|
| Encoder | ~11M |
| Projector | ~12.6M |
| **Total trainable** | **~23.6M** |

Well under the 100M limit.

---

## Dataset: active_matter

Physical simulations of active matter dynamics.

| Property | Value |
|---|---|
| Source | HuggingFace `polymathic-ai/active_matter` |
| Input channels | 11 (concentration, velocity, orientation tensor, strain-rate tensor) |
| Spatial resolution | 224×224 (center-cropped from 256×256) |
| Temporal length | 16 frames per clip |
| Train samples | 8,750 |
| Validation samples | 1,200 |
| Test samples | 1,300 |
| Physical parameters | α (active dipole strength, 5 values), ζ (steric alignment, 9 values) |
| Unique param combos | 45 |

### Data Augmentation (training only)
- Random crop 224×224 from 256×256
- Random horizontal and vertical flip
- Random 90° rotation (0/90/180/270°)
- Gaussian noise (std=1.0)

### Normalization
Per-sample, per-channel z-score normalization across T×H×W.

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 20 |
| Batch size (per GPU) | 4 |
| Effective batch size | 32 (via gradient accumulation) |
| Learning rate | 1e-3 |
| LR schedule | Cosine annealing with linear warmup (5 epochs) |
| Weight decay | 0.05 |
| Gradient clip | 1.0 |
| Optimizer | AdamW |
| Mixed precision | bfloat16 |
| stride | 4 (2,275 training clips) |

---

## Training Objective

Two augmented views of the same 16-frame clip are passed through the shared encoder and projector. The VICReg loss is computed on the projected embeddings (B, 2048). The encoder learns to produce representations that are:
- **Invariant** to augmentations of the same clip
- **Non-collapsed** (each embedding dimension has variance ≥ 1)
- **Decorrelated** (no redundant dimensions)

The projector is purely a training aid — it is thrown away after training. Only the encoder's 384-dim output is used for evaluation.

---

## Evaluation

The frozen encoder (384-dim mean-pooled output) is evaluated by predicting the physical parameters α and ζ using:

1. **Linear probe** — single `nn.Linear(384, 1)` trained with MSE loss
2. **kNN regression** — k=20 nearest neighbors with cosine distance

Both targets are z-score normalized. MSE is reported on the validation and test sets.

---

## Files

| File | Description |
|---|---|
| `model.py` | Full model: PatchEmbed3D, ViTEncoder, ProjectionMLP, VICRegLoss, VICReg |
| `dataset.py` | ActiveMatterDataset — loads HDF5 clips, augmentation, normalization |
| `train.py` | Training loop with gradient accumulation, AMP, checkpointing, W&B logging |
| `evaluate.py` | Linear probe + kNN regression on frozen encoder embeddings |
| `submit.slurm` | SLURM job script for NYU HPC (A100 GPU) |

---

## Running

**Training:**
```bash
python train.py
# Resume from checkpoint:
python train.py --resume /scratch/sb10583/checkpoints/vicreg-v2/latest.pt
```

**Evaluation:**
```bash
python evaluate.py --checkpoint /scratch/sb10583/checkpoints/vicreg-v2/best.pt --test
```

**HPC:**
```bash
sbatch submit.slurm
```
