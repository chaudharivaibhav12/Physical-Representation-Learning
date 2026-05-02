# ViT-JEPA v2 — Vision Transformer JEPA on Active Matter

Self-supervised representation learning on physical simulations using a 3D Vision Transformer JEPA with token-level VICReg loss. This is the ViT-based contribution experiment: a global-attention encoder replaces the ConvNeXt backbone, and VICReg statistics are computed over all 392 tokens per sample rather than only the pooled vector.

---

## Architecture Overview

```
Training:
  context (B, 11, 16, 224, 224)
      → PatchEmbed3D: Conv3D(kernel=(2,32,32), stride=(2,32,32))
      → (B, 392, 384)  [8T × 7H × 7W = 392 tokens]
      → + 3D sinusoidal positional embedding (fixed)
      → ViTEncoder: 6 × TransformerBlock(dim=384, heads=6, mlp=1536)
      → LayerNorm
      → ctx_tokens (B, 392, 384)
      → TransformerPredictor:
            Linear(384→192) → 2×TransformerBlock(dim=192,heads=4) → LayerNorm → Linear(192→384)
      → pred_tokens (B, 392, 384)
      → global avg pool → z_pred (B, 384)
                                           ┐
  target (B, 11, 16, 224, 224)             ├─→ VICReg Loss
      → same Encoder (torch.no_grad())     │    invariance on pooled (B, 384)
      → tgt_tokens (B, 392, 384)           │    variance + covariance on tokens (B×392, 384)
      → global avg pool → z_tgt (B, 384) ┘

Evaluation (predictor discarded):
  clip → Encoder → global avg pool → (B, 384) → Linear Probe / kNN
```

---

## Components

### 1. PatchEmbed3D

Converts a raw clip to tokens with a single Conv3D:

```
Input:  (B, 11, 16, 224, 224)
  → Conv3D(in=11, out=384, kernel=(2,32,32), stride=(2,32,32))
  → reshape → (B, 392, 384)

Token breakdown:
  Temporal: 16 / 2 = 8
  Spatial:  224 / 32 = 7 × 7 = 49
  Total:    8 × 49 = 392
```

### 2. Positional Embedding

Fixed 3D sinusoidal embeddings — not learned. The embed_dim (384) is split equally into three parts:

```
pos_embed: (1, 392, 384)
  = [t_sincos (128 dims) | h_sincos (128 dims) | w_sincos (128 dims)]
```

### 3. ViTEncoder

6-layer pre-norm transformer. Returns the full token sequence (no pooling inside the encoder).

| Property | Value |
|---|---|
| Layers | 6 |
| Embedding dim | 384 |
| Attention heads | 6 |
| Head dim | 64 |
| MLP dim | 1536 (4×) |
| Normalization | Pre-LayerNorm |
| Output | (B, 392, 384) — full token sequence |

`forward()` returns `(B, 392, 384)`. `forward_pooled()` globally average-pools to `(B, 384)`.

### 4. TransformerPredictor

Shallow 2-layer transformer that maps context tokens to predicted target tokens, then pools.

```
ctx_tokens (B, 392, 384)
  → Linear(384 → 192)          [input projection]
  → (B, 392, 192)
  → 2 × TransformerBlock(dim=192, heads=4)
  → LayerNorm
  → Linear(192 → 384)          [output projection]
  → pred_tokens (B, 392, 384)  [predictor output — not yet pooled]
```

Pooling to `z_pred (B, 384)` happens in `ViTJEPA.forward()` after the predictor returns.

### 5. VICReg Loss

Applied between `z_pred` and `z_tgt`. Invariance is computed on the pooled vectors; variance and covariance statistics are computed on the full flat token batch for a more stable estimate.

| Term | Weight | Applied on |
|---|---|---|
| Invariance (sim) | 2.0 | Pooled `(B, 384)` — MSE between predicted and target |
| Variance (std) | 20.0 | Flat tokens `(B×392, 384)` — std hinge ≥ 1 per dim |
| Covariance (cov) | 2.0 | Flat tokens `(B×392, 384)` — decorrelate dimensions |

Variance and covariance statistics are computed in fp32 for numerical stability (upcast from bf16 inputs).

---

## Parameter Count

| Component | Parameters |
|---|---|
| PatchEmbed3D | ~135K |
| ViTEncoder (6 layers, dim=384) | ~5.3M |
| TransformerPredictor (2 layers, dim=192) | ~0.6M |
| **Total trainable** | **~6.0M** |

---

## Dataset: active_matter

Physical simulations of active matter dynamics with physics-aware augmentations.

| Property | Value |
|---|---|
| Source | HuggingFace `polymathic-ai/active_matter` (HDF5) |
| Input channels | 11 (concentration, velocity, orientation tensor, strain-rate tensor) |
| Spatial resolution | 224×224 (cropped from 256×256) |
| Temporal length | 16 frames per clip (context) + 16 frames (target) |
| Train samples | ~8,750 |
| Validation / Test samples | ~1,200 / ~1,300 |
| Physical parameters | α (active dipole strength, 5 values), ζ (steric alignment, 9 values) |
| Unique param combos | 45 |
| Dataset class | `ActiveMatterDataset` (module: `dataset`) |
| Sampling stride | 1 (training; eval uses stride=4) |

### Data Augmentation (training only)

- Random 224×224 crop (from 256×256)
- Random horizontal flip (p=0.5) with per-channel sign correction (`_HFLIP_SIGN`)
- Random vertical flip (p=0.5) with per-channel sign correction (`_VFLIP_SIGN`)
- Gaussian noise (std=1.0)

Sign correction arrays ensure that vector and tensor field channels are physically consistent after spatial flips (e.g., the x-component of velocity flips sign under a horizontal flip).

### Val / Test Preprocessing

Center crop to 224×224; no noise; no flips.

### Normalization

Per-sample, per-channel z-score normalization across T×H×W.

---

## Training Configuration

Values shown are from `train.py CONFIG` dict and `run_vit_jepa_v2_p32.sbatch`.

| Hyperparameter | Value | Source |
|---|---|---|
| Epochs | 100 | CONFIG / SLURM |
| Per-device batch size | 8 | CONFIG / SLURM |
| Target global batch size | 64 | CONFIG |
| Gradient accumulation steps | 8 (= 64 / 8) | computed |
| Learning rate | 1e-3 | CONFIG |
| LR schedule | Cosine decay, 5-epoch warmup, min=1e-6 | CONFIG |
| Weight decay | 0.05 | CONFIG |
| Gradient clip | 1.0 | CONFIG |
| Optimizer | AdamW, betas=(0.9, 0.95) | CONFIG |
| Mixed precision | bfloat16 | CONFIG |
| Sampling stride (train) | 1 | CONFIG / SLURM |
| VICReg sim / std / cov | 2.0 / 20.0 / 2.0 | CONFIG |

---

## Training Objective

Given 16 context frames, the encoder produces token-level features that the predictor maps to match the encoder's output on the next 16 target frames. The target path runs under `torch.no_grad()` — no EMA encoder; stop-gradient is sufficient with VICReg preventing collapse.

VICReg applied at two levels:
- **Invariance** on pooled `(B, 384)` vectors — forces predictive alignment
- **Variance + Covariance** on flat `(B×392, 384)` tokens — 3136 samples per batch (vs 8 for pooled-only), giving a stable covariance estimate that prevents collapse without augmentation pairs

The predictor is a training aid only — discarded after training. Only the encoder's 384-dim pooled output is used for evaluation.

---

## Preemption Handling and Checkpointing

`train.py` registers UNIX signal handlers for preemption-safe saves:

- **SIGUSR1** — sent by SLURM 90 seconds before job end (`--signal=SIGUSR1@90` in sbatch). Triggers an emergency save with `OffsetSampler` state, then re-queues the job.
- **SIGTERM** — fallback; also triggers emergency save.

Saves are atomic: written to a `.tmp` file then `os.replace`d to avoid corrupt checkpoints on hard kills.

**OffsetSampler** records `(epoch, sample_offset)` so that a resumed job skips already-processed samples within the current epoch rather than re-starting from the beginning.

Per-epoch saves:
- `epoch_N.pt` — full training state at epoch N
- `latest.pt` — always the most recent
- `best.pt` — lowest validation loss so far
- Wall-clock saves every 5 minutes (in addition to epoch saves)

Each checkpoint contains: `model`, `optimizer`, `scheduler`, `epoch`, `step_in_epoch`, `best_val_loss`, and full config for reproducibility.

---

## Dataset Interface

```python
from dataset import ActiveMatterDataset

train_ds = ActiveMatterDataset(
    split="train",
    cache_dir="/path/to/active_matter/cache",
    num_frames=16,
    stride=1,
)
# Returns dicts with keys: "context", "target", "alpha", "zeta"
# context / target shape: (11, 16, 224, 224)
```

---

## Files

| File | Description |
|---|---|
| `model.py` | PatchEmbed3D, ViTEncoder, TransformerPredictor, VICRegLoss, ViTJEPA; sanity check in `__main__` |
| `dataset.py` | ActiveMatterDataset with physics-aware flips, OffsetSampler |
| `train.py` | Training loop (DDP + AMP + gradient accumulation, W&B logging, SIGUSR1/SIGTERM handlers) |
| `evaluate.py` | Linear probe + kNN regression on frozen encoder embeddings |
| `run_vit_jepa_v2_p32.sbatch` | SLURM job script for NYU HPC (A100 GPU, 16h wall time, 32 GB RAM) |

---

## Quickstart

### 1. Verify the model builds

```python
from model import ViTJEPA

model = ViTJEPA()
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
```

Or run the built-in sanity check:

```bash
python model.py
```

Expected output:
```
encoder:    5,3XX,XXX
predictor:    6XX,XXX
total:      ~6,000,000
```

### 2. Single-GPU training

```bash
python train.py
```

### 3. Multi-GPU training

```bash
torchrun --nproc_per_node=4 train.py
```

### 4. Submit SLURM job

```bash
sbatch run_vit_jepa_v2_p32.sbatch
```

### 5. Resume from checkpoint

```bash
python train.py --resume /path/to/checkpoints/latest.pt
```

The OffsetSampler state is restored, so training resumes mid-epoch at the exact sample where it was interrupted.

### 6. Evaluate after training

```bash
python evaluate.py --checkpoint /path/to/checkpoints/best.pt \
    --cache-dir /path/to/active_matter/cache
```

---

## Evaluation

The frozen encoder (384-dim global-average-pooled output) is evaluated by predicting α and ζ using:

1. **Linear probe** — separate `nn.Linear(384, 1)` for each target (α and ζ), trained independently with MSE loss on z-score-normalized targets
2. **kNN regression** — `KNeighborsRegressor(k=20, metric="cosine")` with `StandardScaler` applied to embeddings; separate models for α and ζ

Features are extracted with `eval stride=4` (no augmentation). MSE is reported in both normalized and original physical units on val and test sets.

---

## Collapse Monitoring

`train.py` computes `check_collapse()` using `forward_pooled()` after each epoch, logging the average std across embedding dimensions to W&B:

| Avg std | Interpretation |
|---|---|
| > 0.3 | Healthy |
| ~0.1 | Warning — partial collapse |
| < 0.1 | Collapsed — check VICReg weights |

There is no separate `collapse_check.py` — diagnostics are built into the training loop.

---

## Memory & Throughput Notes

Knobs if you hit OOM:
- `batch_size` — drop from 8 to 4; grad accum compensates
- `patch_size` — 32 → 48 (fewer tokens: 392 → 196)
- `embed_dim` — 384 → 256
- `depth` — 6 → 4 encoder layers
- `pred_depth` — 2 → 1 predictor layer

---

## v2 Improvements over v1

| Change | v1 | v2 |
|---|---|---|
| VICReg std_weight | 40 | 20 (halved; token-level stats are better conditioned) |
| VICReg application | Pooled (B, 384) only | Invariance on pooled; var/cov on flat tokens (B×392, 384) |
| Var/cov computation | fp16/bf16 | Upcast to fp32 for numerical stability |
| Batch size | 4 | 8 |
| Target global batch | 256 | 64 (accum_steps=8 instead of 64) |

---

## What's Different from the ConvNeXt Baseline

| Aspect | ConvNeXt (conv-JEPA) | This variant (ViT-JEPA v2) |
|---|---|---|
| Encoder | 5-stage ConvNeXt (3D→2D) | 6-layer ViT with PatchEmbed3D |
| Receptive field | Local (grows per stage) | Global from layer 1 |
| Token/embedding count | 1 vector (128-dim after pool) | 392 tokens (384-dim each) |
| Predictor | Convolutional (Conv→ResBlock→Conv) | 2-layer transformer on token sequence |
| Loss | VICReg on (B, 128, 14, 14) embeddings | VICReg; var/cov on flat (B×392, 384) |
| Collapse prevention | VICReg variance/covariance | VICReg variance/covariance (token-level) |
| Target encoder | No EMA (shared or stop-grad) | Stop-gradient only (no EMA) |
| Linear probe output | `nn.Linear(128, 2)` — joint α+ζ | Separate `nn.Linear(384, 1)` per target |
| kNN distance metric | Euclidean | Cosine (with StandardScaler) |
| Parameters | ~3.3M | ~6.0M |
| SLURM RAM | 32 GB | 32 GB |
| Epochs | 50 (aug+VICReg) / 100 (no-aug) | 100 |

---

## Next Steps

1. Run `evaluate.py` on the best checkpoint to get linear probe and kNN MSE on α and ζ.
2. Compare against ConvNeXt baselines to quantify the benefit of global attention.
3. Ablate `patch_size` (32 vs 16) — 16×16 patches give 1568 tokens with finer spatial resolution but 4× more memory.
4. Ablate `depth` (6 vs 4 vs 8 encoder layers) to find the optimal capacity.
5. Consider adding masking (IJEPA-style random token masking) as a harder prediction target.
