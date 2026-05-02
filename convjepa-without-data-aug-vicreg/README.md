# Conv-JEPA (No Augmentation) — VICReg on Active Matter

Self-supervised representation learning on physical simulations using a convolutional JEPA architecture trained with the VICReg objective and **no data augmentation**. This is the no-augmentation ablation of the conv-JEPA baseline: the architecture and VICReg loss are identical to the aug variant, but the dataset applies no spatial flips, rotations, or Gaussian noise. This isolates the contribution of augmentation to representation quality.

---

## Architecture Overview

```
Training:
  context (B, 11, 16, 224, 224)
      → ConvNeXt Encoder (5 stages, 3D → 2D)
          Stage 0: stem     (B, 11,  16, 224, 224) → (B, 16,  16, 224, 224)  [3 res blocks, no downsample]
          Stage 1: down3d   (B, 16,  16, 224, 224) → (B, 32,   8, 112, 112)  [2× downsample, 3 res blocks]
          Stage 2: down3d   (B, 32,   8, 112, 112) → (B, 64,   4,  56,  56)  [2× downsample, 3 res blocks]
          Stage 3: down3d   (B, 64,   4,  56,  56) → (B, 128,  2,  28,  28)  [2× downsample, 9 res blocks]
          Stage 4: down3d   (B, 128,  2,  28,  28) → (B, 128,  1,  14,  14)  [T collapses → 2D, 3 res blocks]
      → context_latent (B, 128, 14, 14)
      → ConvPredictor: Conv2d(128→256) → ResidualBlock(256) → Conv2d(256→128)
      → predicted_latent (B, 128, 14, 14)
                                           ┐
  target (B, 11, 16, 224, 224)             ├─→ VICReg Loss
      → same Encoder (shared weights)      │    (dense spatial, 14×14=196 vectors/sample)
      → target_latent (B, 128, 14, 14)    ┘

Evaluation (predictor discarded):
  clip → Encoder → global avg pool → (B, 128) → Linear Probe / kNN
```

---

## Components

### 1. ConvNeXt Encoder
5-stage convolutional backbone that collapses the temporal dimension progressively.

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
Lightweight convolutional head that predicts the target latent from the context latent.

```
(B, 128, 14, 14)
  → Conv2d(128 → 256, k=2, pad=1)
  → ResidualBlock(256)              [depthwise 7×7 conv → LayerNorm → 4× MLP → LayerScale]
  → Conv2d(256 → 128, k=2)
→ predicted_latent (B, 128, 14, 14)
```

### 3. VICReg Loss
Applied between predicted and target dense spatial embeddings (B, 128, 14, 14), treated as a bag of 196 vectors per sample.

| Term | Weight | Purpose |
|---|---|---|
| Invariance (sim) | 2.0 | MSE between predicted and target latents |
| Variance (std) | 20.0 | Keep per-dim std ≥ 1 — prevent collapse |
| Covariance (cov) | 2.0 | Decorrelate embedding dimensions |
| Chunks per batch | 5 | Shuffle-and-chunk for stable covariance estimate |

---

## Parameter Count

| Component | Parameters |
|---|---|
| Encoder | ~2.47M |
| Predictor | ~0.80M |
| **Total trainable** | **~3.27M** |

---

## Dataset: active_matter

Physical simulations of active matter dynamics. **No augmentation is applied** — this is the key difference from the aug+VICReg variant.

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
None. `augment=false` and `noise_std=0.0` in config. The VICReg variance and covariance terms are solely responsible for preventing representation collapse.

### Normalization
Per-sample, per-channel z-score normalization across T×H×W.

---

## Training Configuration

Values shown are those used in the actual HPC run (SLURM script overrides take precedence over `config.yaml` defaults where they differ).

| Hyperparameter | Value | Source |
|---|---|---|
| Epochs | 25 | SLURM override |
| Per-device batch size | 8 | config.yaml |
| Target global batch size | 256 (via gradient accumulation) | config.yaml / SLURM |
| Learning rate | 1e-3 | config.yaml |
| LR schedule | Cosine annealing, 1-epoch warmup, min=1e-6 | config.yaml |
| Weight decay | 0.05 | config.yaml |
| Gradient clip | 1.0 | config.yaml |
| Optimizer | AdamW, betas=(0.9, 0.95) | config.yaml |
| Mixed precision | bfloat16 | config.yaml |
| VICReg sim / std / cov | 2 / 20 / 2 | config.yaml |
| VICReg chunks per batch | 5 | config.yaml |

---

## Training Objective

Given 16 context frames, the encoder produces a spatial latent that the predictor maps to match the encoder's latent of the next 16 target frames. VICReg is applied on the dense spatial embeddings (196 vectors per sample from the 14×14 output). The encoder learns representations that are:
- **Predictive** — context latent can be transformed to match the target latent
- **Non-collapsed** — VICReg variance hinge keeps per-dim std ≥ 1
- **Decorrelated** — VICReg covariance term removes redundant dimensions

Without augmentation, the two views (context and target) are temporally adjacent non-augmented clips, so the VICReg loss bears the full burden of collapse prevention.

The predictor is a training aid only — discarded after training. Only the encoder's 128-dim pooled output is used for evaluation.

---

## Dataset Interface

The trainer imports the dataset class dynamically. Your dataset must:

1. Accept `split="train" | "val"` in its constructor (plus any kwargs).
2. Return dicts from `__getitem__` with:
   - `"context"` — tensor of shape `(C, T, H, W)` where `C=11, T=16, H=W=224`
   - `"target"` — tensor of the same shape
3. Apply normalization inside the dataset; augmentation must be disabled.

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
| `loss.py` | VICReg with shuffle-and-chunk |
| `scheduler.py` | Cosine LR with linear warmup |
| `train.py` | Training loop (DDP + AMP + gradient accumulation, W&B logging) |
| `collapse_check.py` | Representation collapse diagnostics (effective rank, channel stats, kNN identity) |
| `config.yaml` | Training config |
| `run_slurm_job_no_aug.sbatch` | SLURM job script for NYU HPC (A100 GPU, 16h wall time, 80 GB RAM) |

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
python train.py --config config.yaml train.lr=5e-4 train.num_epochs=25
```

### 6. Resume from checkpoint

```bash
python train.py --config config.yaml --resume ./checkpoints/<run>/latest.pt
```

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

Each checkpoint contains: `encoder`, `predictor`, `optimizer`, `scheduler`, `epoch`, `global_step`, `best_val_loss`, and the full config for reproducibility.

---

## Collapse Diagnostics

`collapse_check.py` uses `ActiveMatterDatasetNoAug` for feature extraction. With no augmentation, the VICReg std hinge is the only collapse-prevention mechanism, making these diagnostics especially useful to verify the variance term is holding.

```bash
python collapse_check.py \
    --checkpoint /path/to/epoch_N.pt \
    --cache-dir /path/to/active_matter/cache \
    --output-json collapse_epochN.json
```

Reports:
- **Effective rank** — entropy of the singular value spectrum; healthy encoder ≈ close to 128
- **Participation ratio** — how many dimensions carry meaningful energy
- **Dead channel fraction** — channels with std < 1e-4 (should be near 0)
- **Near-unit channel fraction** — channels with std ∈ [0.9, 1.1]; high values indicate the encoder is hugging the VICReg std hinge (partial collapse signal)
- **kNN identity rate** — whether distinct trajectories map to distinct features

---

## Memory & Throughput Notes

Knobs if you hit OOM:
- `train.batch_size` — drop from 8 to 4 or 2; grad accum compensates
- `train.amp_dtype: "bf16"` (default) — keeps activations in bfloat16
- `dataset.num_frames` — dropping from 16 to 8 halves the time dimension throughout
- `dataset.kwargs.resolution: [112, 112]` — halves H×W (4× activation reduction)
- Enable `torch.utils.checkpoint` in res blocks (not wired in here yet)

---

## What's Different from the Other Variants

| Aspect | This variant (no-aug VICReg) | Aug + VICReg | No-aug + EMA |
|---|---|---|---|
| Loss | VICReg | VICReg | Pure MSE |
| Collapse prevention | VICReg variance/covariance | VICReg variance/covariance | EMA target encoder |
| Data augmentation | **None** | Yes (flip, rotate, noise std=1.0) | None |
| Encoders | 1 (shared) | 1 (shared) | 2 (online + EMA target) |
| Dataset class | `ActiveMatterDatasetNoAug` | `ActiveMatterDataset` | `ActiveMatterDatasetNoAug` |
| Eval script | `collapse_check.py` only | `eval_probe.py` + `collapse_check.py` | `eval_probe.py` + `collapse_check.py` |
| Epochs (actual run) | 25 | 50 | 100 |

---

## Next Steps

1. Run linear probe + kNN evaluation on frozen encoder features against the `alpha`/`zeta` regression targets (requires adding `eval_probe.py` from one of the other variants — it loads checkpoints by config, so it is directly reusable).
2. Run `collapse_check.py` to verify effective rank — without augmentation, the VICReg variance term carries the full collapse-prevention burden; near-unit channel fraction is the key diagnostic.
3. Compare evaluation MSE against the aug+VICReg variant to isolate the effect of augmentation.
4. Compare against the no-aug+EMA variant to isolate VICReg vs. EMA as collapse-prevention mechanisms.
