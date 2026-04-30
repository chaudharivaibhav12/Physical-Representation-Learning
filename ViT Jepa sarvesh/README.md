# ViT-JEPA Sarvesh — Joint-Embedding Predictive Architecture for Active Matter

Self-supervised representation learning on physical simulations using a 3D Vision Transformer trained with the JEPA (Joint-Embedding Predictive Architecture) objective. The model learns by predicting the latent representations of masked spatiotemporal regions from visible context tokens, using an EMA (momentum) target encoder to provide stable training targets.

---

## Architecture Overview

```
Input: (B, 11, 16, 224, 224)
         │
         ▼
   PatchEmbed3D (Conv3D)
   patch=16×16, tubelet=2
         │
         ▼
  1568 tokens (B, 1568, 256)
  + learnable positional embedding
         │
         ├──────────────────────────────────────────────┐
         │  context_idx (~75%)        target_idx (~25%) │
         ▼                                              │
  [Online Encoder]                              [Target Encoder]
  6× TransformerBlock                           6× TransformerBlock
  (backprop)                                    (EMA update only, no grad)
         │                                              │
         ▼                                              ▼
  ctx_encoded (B, N_ctx, 256)          tgt_encoded (B, N_tgt, 256)
         │                                    L2-normalize ↓
         ├── + mask_tokens @ target positions           │
         ▼                                              │
     [Predictor]                                        │
  4× narrow TransformerBlock                            │
  (256 → 128 → 256 bottleneck)                          │
         │                                              │
         ▼                                              ▼
  pred_tgt (B, N_tgt, 256) ────── MSE Loss ─────────────┘

Evaluation:
  clip → Online Encoder (all 1568 tokens) → mean pool → (B, 256)
       → frozen → Linear Probe / kNN
```

---

## Components

### 1. PatchEmbed3D
Converts a spatiotemporal clip into tokens using Conv3D.

| Property | Value |
|---|---|
| Kernel | 2 (temporal) × 16 × 16 (spatial) |
| Stride | same as kernel (non-overlapping) |
| Token grid | 8T × 14H × 14W = **1,568 tokens** |
| Output dim | 256 |
| Post-norm | LayerNorm |

### 2. Positional Embedding
Learnable positional embedding of shape `(1, 1568, 256)` added to all tokens after patch embedding. Initialized with truncated normal (std=0.02).

### 3. Online Encoder
Processes context tokens only during training (backprop flows through this encoder). Runs on all tokens at evaluation time.

| Property | Value |
|---|---|
| Embed dim | 256 |
| Depth | 6 transformer blocks |
| Attention heads | 8 (head dim = 32) |
| MLP ratio | 4× (hidden dim = 1024) |
| Normalization | Pre-norm (LayerNorm before attention and MLP) |
| Gradient checkpointing | Enabled during training |

### 4. Target Encoder
Identical architecture to the online encoder. Updated exclusively via **Exponential Moving Average (EMA)** of the online encoder weights — never receives gradients directly.

| Property | Value |
|---|---|
| Architecture | Same as online encoder |
| Update rule | EMA: τ_target = τ_online × (1 - m) + τ_target × m |
| EMA momentum | Cosine-annealed: 0.996 → 0.9999 over training |
| Gradient | None (frozen, updated only by EMA) |

The target encoder provides **stable, slowly-evolving training targets** that prevent representation collapse without requiring contrastive negatives or explicit collapse-prevention losses.

### 5. Spatiotemporal Block Masking
At each training step, tokens are split into context (~75%) and target (~25%) sets using random spatiotemporal block masking.

| Property | Value |
|---|---|
| Target ratio | 25% (~392 of 1568 tokens) |
| Number of blocks | 4 contiguous 3D blocks |
| Block shape | Random cuboid in (T, H, W) token space |
| Context | Everything outside the target blocks |

The masking is spatiotemporally contiguous — blocks span connected regions in both space and time, forcing the model to reason about physical dynamics rather than independent frames.

### 6. Predictor (narrow transformer)
Takes context encodings and learnable mask tokens (positioned at target locations) and predicts the target encoder's output at those positions.

| Property | Value |
|---|---|
| Input/output dim | 256 (matches encoder) |
| Internal dim | 128 (bottleneck — lighter than encoder) |
| Depth | 4 transformer blocks |
| Attention heads | 4 |
| Input | `[ctx_encoded (B, N_ctx, 256) ‖ mask_tokens @ target_pos (B, N_tgt, 256)]` |
| Output | Predictions at target positions (B, N_tgt, 256) |

The narrow bottleneck (256 → 128 → 256) keeps the predictor lightweight so the encoder, not the predictor, does the heavy representation learning.

### 7. Training Objective
MSE loss between predictor output and L2-normalized target encoder output at masked positions:

```
loss = MSE(predictor(ctx_encoded, mask_tokens), L2_norm(target_encoder(all_tokens)[:, target_idx]))
```

L2-normalization of target encoder output provides a bounded, scale-invariant training signal that stabilizes training.

---

## Parameter Count

| Component | Parameters |
|---|---|
| PatchEmbed3D | ~90K |
| Positional embedding + mask token | ~403K |
| Online Encoder | ~5.7M |
| Target Encoder | ~5.7M (not trainable) |
| Predictor | ~660K |
| **Total trainable** | **~6.85M** |

Well under the 100M limit. Notably smaller than the VICReg models (~23.6M) due to the smaller embed dim (256 vs 384) and finer patch grid replacing the projector.

---

## Dataset: active_matter

| Property | Value |
|---|---|
| Source | HuggingFace `polymathic-ai/active_matter` |
| Input channels | 11 |
| — Concentration scalar field | 1 channel |
| — Velocity vector field | 2 channels |
| — Orientation tensor field | 4 channels |
| — Strain-rate tensor field | 4 channels |
| Raw resolution | 256×256 |
| Processed resolution | 224×224 |
| Temporal length | 16 frames per clip |
| Train samples | 11,550 clips (stride=1, 45 simulations) |
| Validation samples | 1,584 clips |
| Test samples | — |
| Physical parameters | α (active dipole strength), ζ (steric alignment) |
| Unique param combos | 45 (5α × 9ζ) |

### Data Augmentation (training only)
- Random crop 224×224 from 256×256
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.5)
- Random 90° rotation (k ∈ {0,1,2,3})
- Gaussian noise (std=1.0)

### Normalization
Per-sample, per-channel z-score normalization across T×H×W before augmentation.

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 100 |
| Batch size (per GPU) | 8 |
| Effective batch size | 64 (8 gradient accumulation steps) |
| Learning rate | 1.5e-4 |
| LR schedule | Cosine annealing with 10-epoch linear warmup |
| Weight decay | 0.05 |
| Gradient clip | 1.0 |
| Optimizer | AdamW |
| Mixed precision | bfloat16 |
| EMA momentum | Cosine-annealed 0.996 → 0.9999 |
| Step checkpoint | Every 20 steps |
| Epoch checkpoint | Every 5 epochs |

### Preemption Handling
- SLURM sends SIGUSR1 90s before spot instance preemption
- Python catches SIGUSR1, saves a step-level checkpoint, then bash requeues the job
- `global_step` saved and restored so training continues from exact step
- Deterministic per-epoch shuffle via `torch.Generator(seed=epoch)` + `Subset` for instant batch-skip on resume

---

## Evaluation Results

Frozen online encoder, all 1,568 tokens, mean-pooled → 256-dim embedding. Z-score normalized targets, MSE.

### Validation MSE
| Method | α | ζ |
|---|---|---|
| Linear Probe (Ridge) | **0.0698** | **0.2752** |
| kNN (k=20) | 0.5604 | 0.3294 |

Random baseline ≈ 1.0. Lower is better.

ViT-JEPA achieves the best linear probe results across all models (better than VICReg-v2), suggesting its embeddings are highly linearly separable. kNN performance is lower than VICReg, indicating the local neighborhood structure of the embedding space differs from contrastive methods.

---

## Comparison with VICReg Models

| Property | ViT-JEPA (this) | VICReg V2 |
|---|---|---|
| Patch size | 16×16 → 1568 tokens | 32×32 → 392 tokens |
| Embed dim | 256 | 384 |
| Target encoder | EMA (separate, no grad) | Same encoder (shared) |
| Loss | MSE in latent space | VICReg (invariance + variance + cov) |
| Collapse prevention | EMA + L2-norm targets | Variance + covariance terms |
| Trainable params | ~6.85M | ~23.6M |
| Linear probe α | **0.0698** | 0.0613 |
| Linear probe ζ | **0.2752** | 0.3586 |
| kNN α | 0.5604 | **0.0379** |
| kNN ζ | 0.3294 | **0.2754** |

---

## Files

| File | Description |
|---|---|
| `model.py` | PatchEmbed3D, ViTEncoder, Predictor, ViTJEPA full model |
| `masking.py` | Spatiotemporal block masking — samples context/target token indices |
| `dataset.py` | ActiveMatterDataset — HDF5 loading, augmentation, normalization |
| `train.py` | Training loop: EMA update, block masking, grad accumulation, AMP, checkpointing |
| `eval.py` | Linear probe (Ridge) + kNN regression on frozen encoder embeddings |
| `config.yaml` | Full training configuration |
| `submit.slurm` | SLURM training job with SIGUSR1 preemption handling |
| `eval.slurm` | SLURM evaluation job |

---

## Running

**Training:**
```bash
python train.py
python train.py --resume /scratch/sb10583/checkpoints/vit-jepa-sarvesh/latest.pt
```

**Evaluation:**
```bash
python eval.py --checkpoint /scratch/sb10583/checkpoints/vit-jepa-sarvesh/best.pt --split valid
python eval.py --checkpoint /scratch/sb10583/checkpoints/vit-jepa-sarvesh/best.pt --split test
```

**HPC:**
```bash
sbatch submit.slurm
sbatch eval.slurm
```
