# ViT-JEPA for Active Matter Physical Simulations

A 3D Vision Transformer (ViT) based Joint-Embedding Predictive Architecture for learning representations of spatiotemporal physical simulations from the **active_matter** dataset (The Well).

---

## Overview

This implementation trains a self-supervised encoder on physical fluid simulations **without using any labels** (α or ζ). The learned representations are evaluated by how well a frozen encoder's embeddings can predict the underlying physical parameters α and ζ using a linear layer and kNN regression.

### Key Design Choices

| Choice | Reason |
|--------|--------|
| ViT encoder (not CNN) | Global attention captures large-scale flow patterns that determine α and ζ |
| Shallow transformer predictor | Attends over all 392 tokens before predicting — better than a conv/linear head |
| 32×32 spatial patches | Memory-safe: 392 tokens vs 1568 for 16×16 patches |
| VICReg loss | Prevents collapse without needing an EMA target encoder |
| Temporal context→target split | Context = frames 0–15, Target = frames 16–31 |

---

## Architecture

### Full Training Flow

```
context frames (B, 11, 16, 224, 224)
        ↓
   PatchEmbed3D          ← Conv3D(kernel=2×32×32)
        ↓
  (B, 392, 384)          ← 8 time × 7×7 spatial = 392 tokens
        ↓
+ 3D Sinusoidal Pos Embed
        ↓
  ViT Encoder × 6 blocks ← deep, learns physics
        ↓
  (B, 392, 384)          ← full token sequence
        ↓
  Transformer Predictor  ← shallow, 2 blocks only
  (project → attend → project back)
        ↓
  global avg pool
        ↓
  z_pred (B, 384)        ← predicted target embedding
        ↕  VICReg Loss
  z_tgt  (B, 384)        ← pooled target encoder output
        ↑
  global avg pool
        ↑
  (B, 392, 384)
        ↑
  SAME ViT Encoder       ← shared weights, both paths
        ↑
  (B, 392, 384)
        ↑
  PatchEmbed3D
        ↑
target frames (B, 11, 16, 224, 224)
```

### 1. Patch Embedding (`PatchEmbed3D`)

Converts a raw clip into tokens using a single `Conv3D`:

```
Input:  (B, 11, 16, 224, 224)
         ↓  Conv3D(kernel=(2, 32, 32), stride=(2, 32, 32))
Tokens: (B, 392, 384)

Breakdown:
  Temporal patches: 16 / 2  = 8
  Spatial patches:  224 / 32 = 7 × 7 = 49
  Total tokens:     8 × 49  = 392
```

### 2. Positional Embedding

Fixed **3D sinusoidal positional embeddings** (not learned). The embed_dim (384) is split equally into three parts for the temporal, height, and width axes:

```
pos_embed shape: (1, 392, 384)
  = [t_sincos (128 dims) | h_sincos (128 dims) | w_sincos (128 dims)]
```

### 3. ViT Encoder (6 layers, deep)

Standard pre-norm transformer. Returns the **full token sequence** (before pooling) so the predictor gets rich spatiotemporal structure:

```
Tokens + pos_embed: (B, 392, 384)
        ↓
┌───────────────────────────────────────┐
│  TransformerBlock × 6                 │
│                                       │
│  LayerNorm                            │
│  Multi-Head Attention (6 heads)       │  ← every token attends to all 392
│  LayerNorm                            │
│  MLP (384 → 1536 → 384, GELU)         │
└───────────────────────────────────────┘
        ↓
LayerNorm
        ↓
(B, 392, 384)   ← full token sequence, passed to predictor
```

### 4. Shallow Transformer Predictor (2 layers, lightweight)

Takes the context token sequence and predicts what the target embedding should look like. Intentionally shallow — the encoder does the heavy physics learning:

```
Context tokens: (B, 392, 384)
        ↓
input_proj: Linear(384 → 192)     ← project to smaller predictor dim
        ↓
(B, 392, 192)
        ↓
┌───────────────────────────────────────┐
│  TransformerBlock × 2                 │
│  (dim=192, heads=4)                   │  ← attends over all 392 positions
└───────────────────────────────────────┘
        ↓
LayerNorm
        ↓
output_proj: Linear(192 → 384)    ← project back to encoder dim
        ↓
(B, 392, 384)
        ↓
global avg pool → (B, 384)        ← z_pred
```

**Why full token sequence (not pooled vector)?**
- A pooled vector loses all spatial and temporal structure
- The predictor needs to know *where* each part of the future simulation should be
- Operating on 392 tokens lets the predictor attend over specific spatial regions and time steps before summarizing

**Why shallow (2 layers)?**
- The encoder (6 layers) should do the heavy physics learning
- If the predictor is too powerful it shortcircuits the encoder — can predict future states without the encoder learning meaningful representations
- 2 layers is enough to refine the prediction without overpowering the encoder

### 5. VICReg Loss

Applied between `z_pred` (predictor output) and `z_tgt` (pooled target encoder output):

```
loss = 2  × Invariance   +  40 × Variance   +  2 × Covariance

Invariance:  MSE(z_pred, z_tgt)
             "predict the future correctly"

Variance:    mean(max(0, 1 - std_d(z)))  for each dimension d
             "don't collapse — keep all dimensions active"

Covariance:  sum of squared off-diagonal elements of Cov(z) / D
             "don't learn redundant dimensions"
```

The high std_weight (40) is intentional — with only 8,750 training samples, collapse is the primary risk.

---

## Parameter Count

| Component | Parameters |
|-----------|-----------|
| PatchEmbed3D | ~135K |
| ViT Encoder (6 layers, dim=384) | ~5.3M |
| Transformer Predictor (2 layers, dim=192) | ~0.6M |
| LayerNorm + misc | ~5K |
| **Total** | **~6.0M** |

Well under the 100M parameter limit. No pretrained weights — trained from scratch.

---

## File Structure

```
vit_jepa/
├── model.py          # PatchEmbed3D, ViTEncoder, TransformerPredictor, VICRegLoss, ViTJEPA
├── dataset.py        # ActiveMatterDataset with sliding window
├── train.py          # Training loop (DDP + AMP + grad accum + wandb)
├── evaluate.py       # Linear probe + kNN evaluation
├── train.sh          # Slurm batch job script (A100 GPU)
├── sanity_check.py   # Shape and gradient verification
└── README.md         # This file
```

---

## Data Interface

The dataset returns dicts:

```python
{
  "context": Tensor(11, 16, 224, 224),   # frames  0–15,  shape (C, T, H, W)
  "target":  Tensor(11, 16, 224, 224),   # frames 16–31,  shape (C, T, H, W)
  "alpha":   Tensor(scalar),             # NOT used during training
  "zeta":    Tensor(scalar),             # NOT used during training
}
```

### Sliding Window Sampling

Each HDF5 file contains 3 simulations of 81 time steps each. Samples are extracted using a sliding window of 32 frames (16 context + 16 target) with stride 4:

```
Simulation (81 frames):
f0  f1  f2 ... f15 f16 ... f31 f32 ... f80
└──────────────────────────────┘
         Window 1 (32 frames)
    context: f0  → f15
    target:  f16 → f31

    └──────────────────────────────┘
           Window 2 (stride=4)
    context: f4  → f19
    target:  f20 → f35
```

Total: 45 files × 3 sims × ~12 windows ≈ **8,750 training samples** ✓

### Preprocessing Pipeline

```
Raw clip (32, 11, 256, 256)
        ↓
Random 224×224 crop (train) / center crop (val, test)
        ↓
Per-sample, per-channel z-score normalization
        ↓
Gaussian noise augmentation std=1.0  (train only)
        ↓
Split: context = frames 0–15, target = frames 16–31
        ↓
Permute to (C, T, H, W) for Conv3D
```

---

## Training Recipe

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW, betas=(0.9, 0.95) |
| Learning rate | 1e-3 |
| Weight decay | 0.05 |
| LR schedule | Cosine decay, 5-epoch warmup, min=1e-6 |
| Gradient clip | 1.0 |
| Per-device batch size | 4 |
| Target global batch size | 256 (via gradient accumulation) |
| Epochs | 100 |
| Mixed precision | bf16 |
| VICReg sim / std / cov | 2 / 40 / 2 |

---

## Quickstart

### 1. Verify shapes and gradients
```bash
cd /scratch/ok2287/vit_jepa
python sanity_check.py
```

Expected:
```
[PARAMS] encoder:    5,3XX,XXX
[PARAMS] predictor:    6XX,XXX
[PARAMS] total:      ~6,000,000
[PARAMS] < 100M:     True

[SHAPE] (2, 11, 16, 224, 224) → (2, 392, 384)   patch embed
[SHAPE] (2, 392, 384)                             encoder output
[SHAPE] (2, 384)                                  predictor output
[SHAPE] (2, 384)                                  pooled target

[FORWARD] loss: X.XXXX
[BACKWARD] Encoder has gradients:   True
[BACKWARD] Predictor has gradients: True

✓ All checks passed!
```

### 2. Test dataset
```bash
python dataset.py
```

### 3. Dry run (1 epoch, no wandb)
```bash
python train.py --dry-run
```

### 4. Submit training job
```bash
sbatch train.sh
```

### 5. Evaluate after training
```bash
python evaluate.py --checkpoint /scratch/ok2287/checkpoints/vit_jepa/best.pt
```

### 6. Resume if preempted
```bash
python train.py --resume /scratch/ok2287/checkpoints/vit_jepa/latest.pt
```

---

## Evaluation Protocol

As required by the project spec — frozen encoder only, no finetuning.

### Linear Probe
Single `nn.Linear(384 → 1)` trained on frozen embeddings to predict α or ζ. Targets z-score normalized. MSE loss.

### kNN Regression
`KNeighborsRegressor(k=20, metric="cosine")` on frozen embeddings. Targets z-score normalized.

### Collapse Monitoring
After each epoch, embedding std is logged. If std < 0.1, representations are collapsing:

```
std ≈ 0.0  →  COLLAPSED  ← stop training, check VICReg weights
std ≈ 0.1  →  WARNING
std > 0.3  →  HEALTHY
```

---

## Memory Notes

If you hit OOM on the A100:

| Knob | Change |
|------|--------|
| `batch_size` | 4 → 2 (grad accum compensates) |
| `patch_size` | 32 → 48 (fewer tokens: 392 → 196) |
| `embed_dim` | 384 → 256 |
| `depth` | 6 → 4 encoder layers |
| `pred_depth` | 2 → 1 predictor layer |

---

## Why ViT + Predictor over ConvNeXt Baseline?

| Property | ConvNeXt Baseline | This Work (ViT + Predictor) |
|----------|-------------------|-----------------------------|
| Receptive field | ~93 pixels (local) | Full 224×224 (global, layer 1) |
| Temporal collapse | Stage 4 (early) | Only at final pool (late) |
| Prediction head | Conv head (C→2C→C) on pooled vector | 2-layer transformer on full 392 tokens |
| Spatial structure in prediction | Lost after pooling | Preserved through predictor |
| Long-range dependencies | Limited | Full cross-token attention |
| Parameters | ~3.3M | ~6.0M |

The core advantage: α and ζ manifest as **global spatiotemporal patterns**. The ViT encoder captures these globally from layer 1, and the transformer predictor refines the prediction while preserving spatial structure — something neither a ConvNeXt encoder nor a conv/linear predictor can do.