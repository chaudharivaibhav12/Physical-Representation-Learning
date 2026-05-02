# Conv-JEPA with EMA Target Encoder — No Augmentation

Self-supervised representation learning on physical simulations using a convolutional JEPA architecture with an Exponential Moving Average (EMA) target encoder and pure MSE loss. This is the EMA variant of the conv-JEPA baseline: collapse prevention comes from the EMA target (I-JEPA/BYOL style) rather than VICReg, and no data augmentation is applied.

---

## Architecture Overview

```
Training:
  context (B, 11, 16, 224, 224)
      → ConvNeXt Online Encoder (5 stages, 3D → 2D)
          Stage 0: stem     (B, 11,  16, 224, 224) → (B, 16,  16, 224, 224)  [3 res blocks, no downsample]
          Stage 1: down3d   (B, 16,  16, 224, 224) → (B, 32,   8, 112, 112)  [2× downsample, 3 res blocks]
          Stage 2: down3d   (B, 32,   8, 112, 112) → (B, 64,   4,  56,  56)  [2× downsample, 3 res blocks]
          Stage 3: down3d   (B, 64,   4,  56,  56) → (B, 128,  2,  28,  28)  [2× downsample, 9 res blocks]
          Stage 4: down3d   (B, 128,  2,  28,  28) → (B, 128,  1,  14,  14)  [T collapses → 2D, 3 res blocks]
      → ctx_embed (B, 128, 14, 14)
      → ConvPredictor: Conv2d(128→256) → ResidualBlock(256) → Conv2d(256→128)
      → predicted_embed (B, 128, 14, 14)
                                                  ┐
  target (B, 11, 16, 224, 224)                    ├─→ MSE Loss (stop-gradient on target)
      → EMA Target Encoder (θ_t, no grad)         │
      → tgt_embed (B, 128, 14, 14)               ┘

  After each optimizer step:
      θ_t ← τ·θ_t + (1−τ)·θ_online      [EMA update, τ cosine-scheduled 0.996 → 0.9999]

Evaluation (predictor discarded):
  clip → Online Encoder → global avg pool → (B, 128) → Linear Probe / kNN
```

---

## Components

### 1. ConvNeXt Encoder
5-stage convolutional backbone shared by both the online and EMA target branches.

| Property | Value |
|---|---|
| Channel dims | [16, 32, 64, 128, 128] |
| Residual blocks per stage | [3, 3, 3, 9, 3] |
| Convolutions (stages 0–3) | 3D (temporal + spatial) |
| Convolutions (stage 4) | 2D (temporal dim collapsed after squeeze) |
| Downsampling | 2× per stage via stride-2 Conv3d |
| Output spatial | 14×14 |
| Output channels | 128 |
| Normalization | LayerNorm per block (channels_first) |

### 2. ConvPredictor
Lightweight convolutional head that maps the online context embedding to predict the EMA target embedding.

```
(B, 128, 14, 14)
  → Conv2d(128 → 256, k=2, pad=1)
  → ResidualBlock(256)              [depthwise 7×7 conv → LayerNorm → 4× MLP → LayerScale]
  → Conv2d(256 → 128, k=2)
→ predicted_embed (B, 128, 14, 14)
```

### 3. EMA Target Encoder
A non-trainable copy of the online encoder updated after each optimizer step.

| Property | Value |
|---|---|
| Initial momentum τ | 0.996 (target updates 0.4% toward online per step) |
| Final momentum τ | 0.9999 (nearly frozen by end of training) |
| Schedule | Cosine: τ(t) = τ_end − (τ_end − τ_start)·(cos(πt/T)+1)/2 |
| Gradient | None — never trained by backprop |
| Update rule | θ_t ← τ·θ_t + (1−τ)·θ_online, once per optimizer step |

### 4. Loss
Pure MSE between predictor output and the stop-gradient EMA target embedding.

| Property | Value |
|---|---|
| Function | `F.mse_loss(predicted_embed, stop_grad(tgt_embed))` |
| Applied over | Full (B, 128, 14, 14) spatial map |
| Collapse prevention | EMA target (no VICReg variance/covariance terms needed) |

---

## Parameter Count

| Component | Parameters |
|---|---|
| Online Encoder | ~2.47M |
| EMA Target Encoder | ~2.47M (not trained, not counted toward budget) |
| Predictor | ~0.80M |
| **Total trainable** | **~3.27M** |

---

## Dataset: active_matter

Physical simulations of active matter dynamics. **No augmentation is applied** — the EMA target encoder provides collapse prevention instead.

| Property | Value |
|---|---|
| Source | HuggingFace `polymathic-ai/active_matter` |
| Input channels | 11 (concentration, velocity, orientation tensor, strain-rate tensor) |
| Spatial resolution | 224×224 |
| Temporal length | 16 frames per clip (context) + 16 frames (target) |
| Train samples | 8,750 |
| Validation samples | 1,200 |
| Test samples | 1,300 |
| Physical parameters | α (active dipole strength, 5 values), ζ (steric alignment, 9 values) |
| Unique param combos | 45 |
| Dataset class | `ActiveMatterDatasetNoAug` (module: `active_matter_dataset_no_aug`) |

### Data Augmentation
None. `augment=false` and `noise_std=0.0` in config. This is the primary ablation axis vs. the VICReg variant.

### Normalization
Per-sample, per-channel z-score normalization across T×H×W.

---

## Training Configuration

Values shown are those used in the actual HPC run (SLURM script overrides take precedence over `config.yaml` defaults where they differ).

| Hyperparameter | Value | Source |
|---|---|---|
| Epochs | 100 | SLURM override |
| Per-device batch size | 8 | config.yaml |
| Target global batch size | 128 (via gradient accumulation) | SLURM override |
| Learning rate | 1e-3 | config.yaml |
| LR schedule | Cosine annealing, 1-epoch warmup, min=1e-6 | config.yaml |
| Weight decay | 0.05 | config.yaml |
| Gradient clip | 1.0 | config.yaml |
| Optimizer | AdamW, betas=(0.9, 0.95) | config.yaml |
| Mixed precision | bfloat16 | config.yaml |
| EMA momentum start (τ_start) | 0.996 | config.yaml |
| EMA momentum end (τ_end) | 0.9999 | config.yaml |

---

## Training Objective

Given 16 context frames, the online encoder produces a context embedding that the predictor maps to predict the EMA target encoder's embedding of the next 16 target frames. The encoder learns representations that are:
- **Predictive** — context latent can be transformed to match the target latent
- **Non-collapsed** — the slowly-moving EMA target provides a stable, non-trivial prediction target that prevents collapse without explicit variance/covariance constraints
- **Consistent** — increasing τ over training makes the target progressively more stable, encouraging long-range representational consistency

The predictor is a training aid only — discarded after training. Only the online encoder's 128-dim pooled output is used for evaluation.

---

## Dataset Interface

The trainer imports the dataset class dynamically. Your dataset must:

1. Accept `split="train" | "val"` in its constructor (plus any kwargs).
2. Return dicts from `__getitem__` with:
   - `"context"` — tensor of shape `(C, T, H, W)` where `C=11, T=16, H=W=224`
   - `"target"` — tensor of the same shape
3. Apply normalization inside the dataset; augmentation should be disabled.

Point the config at your dataset module:

```yaml
dataset:
  module: "active_matter_dataset_no_aug"
  class_name: "ActiveMatterDatasetNoAug"
  kwargs:
    num_frames: 16
    augment: false
    noise_std: 0.0
```

---

## Files

| File | Description |
|---|---|
| `model.py` | ConvEncoder, ConvPredictor, ResidualBlock, LayerNorm |
| `loss.py` | EMA loss — pure MSE between predictor output and EMA target embedding |
| `scheduler.py` | Cosine LR with linear warmup |
| `train.py` | Training loop with EMA update (DDP + AMP + gradient accumulation, W&B logging) |
| `eval_probe.py` | Linear probe + kNN regression on frozen online encoder embeddings |
| `collapse_check.py` | Representation collapse diagnostics (effective rank, channel stats, kNN identity) |
| `config.yaml` | Training config with EMA momentum parameters |
| `run_slurm_conv_ema.sbatch` | SLURM job script for NYU HPC (A100 GPU, 36h wall time, 80 GB RAM) |

---

## Quickstart

### 1. Verify the model builds

```python
from model import build_jepa, count_params
from omegaconf import OmegaConf

cfg = OmegaConf.load("config.yaml")
encoder, predictor = build_jepa(cfg)
print(f"encoder:   {count_params(encoder):,}")
print(f"predictor: {count_params(predictor):,}")
```

Expected output:
```
encoder:   2,468,720
predictor: 801,664
```

### 2. Edit the config

Set `dataset.module` and `dataset.kwargs` to point at your `ActiveMatterDatasetNoAug`. Also set:
- `out_path` — where checkpoints go
- `run_name` — W&B run name and subdir under `out_path`
- `wandb_project` — W&B project to log to
- `ema_momentum_start` / `ema_momentum_end` — primary ablation knobs for this variant

### 3. Single-GPU training

```bash
python train.py --config config.yaml
```

### 4. Multi-GPU training

```bash
torchrun --nproc_per_node=4 train.py --config config.yaml
```

The trainer auto-detects distributed mode from environment variables set by torchrun. Gradient accumulation automatically adjusts to reach `train.target_global_batch_size`.

### 5. Config overrides from the CLI

OmegaConf dotlist syntax:

```bash
python train.py --config config.yaml train.ema_momentum_start=0.99 train.num_epochs=100
```

### 6. Resume from checkpoint

```bash
python train.py --config config.yaml --resume ./checkpoints/<run>/latest.pt
```

Both the online encoder and EMA target encoder states are restored from the checkpoint.

### 7. Dry run (smoke test without W&B)

```bash
python train.py --config config.yaml --dry-run train.num_epochs=1
```

---

## Checkpoints

Per epoch (subject to `save_every`), the trainer saves:
- `epoch_N.pt` — full training state at epoch N
- `latest.pt` — always the most recent
- `best.pt` — lowest validation loss so far

Each checkpoint contains: `encoder`, `target_encoder`, `predictor`, `optimizer`, `scheduler`, `epoch`, `global_step`, `best_val_loss`, and the full config for reproducibility.

`target_encoder` is saved so that resuming training continues with the exact EMA state rather than reinitializing from a copy of the online encoder.

---

## Evaluation

The frozen **online** encoder (128-dim global-average-pooled output) is evaluated by predicting the physical parameters α and ζ using:

1. **Linear probe** — single `nn.Linear(128, 2)` trained with MSE loss, predicting α and ζ simultaneously
2. **kNN regression** — sweeps k∈{1,3,5,10,20}, selects best k by validation MSE, uses Euclidean distance with inverse-distance weighting

Both targets are z-score normalized using fixed stats (α: mean=−3.0, std=1.414; ζ: mean=9.0, std=5.164). Features are aggregated per trajectory (mean-pooled across windows) before fitting. MSE is reported in both normalized and original physical units on val and test sets.

```bash
python eval_probe.py \
    --checkpoint /path/to/best.pt \
    --cache-dir /path/to/active_matter/cache \
    --output-json results.json
```

---

## Collapse Diagnostics

With EMA-based training there is no explicit variance hinge, so collapse diagnostics are especially important to run. `collapse_check.py` uses `ActiveMatterDatasetNoAug` for feature extraction.

```bash
python collapse_check.py \
    --checkpoint /path/to/epoch_N.pt \
    --cache-dir /path/to/active_matter/cache \
    --output-json collapse_epochN.json
```

Reports:
- **Effective rank** — entropy of the singular value spectrum; healthy encoder ≈ close to 128
- **Participation ratio** — how many dimensions carry meaningful energy
- **Dead channel fraction** — channels with std < 1e-4 (should be near 0; unlike VICReg there is no hinge enforcing this)
- **Near-unit channel fraction** — not directly diagnostic here (no VICReg hinge), but useful for cross-variant comparison
- **kNN identity rate** — whether distinct trajectories map to distinct features

---

## Memory & Throughput Notes

The EMA target encoder doubles memory relative to a single-encoder baseline — two full encoder copies live on GPU throughout training. The SLURM script requests 80 GB RAM (vs 32 GB for the VICReg variant) to account for this.

Knobs if you hit OOM:
- `train.batch_size` — drop from 8 to 4 or 2; grad accum compensates
- `train.amp_dtype: "bf16"` (default) — keeps activations in bfloat16
- `dataset.num_frames` — dropping from 16 to 8 halves the time dimension throughout
- `dataset.kwargs.resolution: [112, 112]` — halves H×W (4× activation reduction)
- Enable `torch.utils.checkpoint` in res blocks (not wired in here yet)

---

## What's Different from the VICReg Variant

| Aspect | VICReg variant | This variant (EMA) |
|---|---|---|
| Loss | VICReg (MSE + variance + covariance terms) | Pure MSE |
| Collapse prevention | VICReg variance/covariance hinge | EMA target encoder |
| Data augmentation | Yes (flip, rotate, Gaussian noise std=1.0) | **None** |
| Encoders during training | 1 (shared for context + target) | 2 (online + EMA target) |
| Checkpoint extras | — | `target_encoder` state dict |
| Dataset class | `ActiveMatterDataset` | `ActiveMatterDatasetNoAug` |
| GPU memory request | 32 GB | 80 GB |

---

## Next Steps

1. Run linear probe + kNN evaluation on frozen encoder features against the `alpha`/`zeta` regression targets.
2. Run `collapse_check.py` to verify effective rank — without VICReg's variance hinge, collapse is possible if EMA momentum is misconfigured.
3. Ablate `ema_momentum_start` and `ema_momentum_end` — these are the primary hyperparameters unique to this variant.
4. Compare evaluation MSE directly against the VICReg variant to isolate the effect of EMA vs. VICReg collapse prevention.
