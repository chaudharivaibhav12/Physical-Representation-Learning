# Conv-JEPA (with Data Augmentation) — VICReg on Active Matter

Self-supervised representation learning on physical simulations using a convolutional JEPA architecture trained with the VICReg objective. Follows the reference conv-JEPA baseline from Qu et al., "Representation Learning for Spatiotemporal Physical Systems."

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
      → same Encoder (no grad)             │    (dense spatial, 14×14=196 vectors/sample)
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
| Downsampling | 2× per stage (stride conv) |
| Output spatial | 14×14 |
| Output channels | 128 |
| Normalization | LayerNorm per block |

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

Matches the reference baseline from the paper.

---

## Dataset: active_matter

Physical simulations of active matter dynamics.

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

### Data Augmentation (training only)
- Random horizontal and vertical flip
- Random 90° rotation (0/90/180/270°)
- Gaussian noise (std=1.0)

### Normalization
Per-sample, per-channel z-score normalization across T×H×W.

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 50 |
| Per-device batch size | 8 |
| Target global batch size | 128 (via gradient accumulation) |
| Learning rate | 1e-3 |
| LR schedule | Cosine annealing, 1-epoch warmup, min=1e-6 |
| Weight decay | 0.05 |
| Gradient clip | 1.0 |
| Optimizer | AdamW, betas=(0.9, 0.95) |
| Mixed precision | bfloat16 |
| VICReg sim / std / cov | 2 / 20 / 2 |
| VICReg chunks per batch | 5 |

---

## Training Objective

Given 16 context frames, the encoder produces a spatial latent that the predictor maps to match the encoder's latent of the next 16 target frames. VICReg is applied on the dense spatial embeddings (196 vectors per sample from the 14×14 output). The encoder learns to produce representations that are:
- **Predictive** — context latent can be transformed to match the target latent
- **Non-collapsed** — each embedding dimension has variance ≥ 1
- **Decorrelated** — no redundant dimensions

The predictor is purely a training aid — it is thrown away after training. Only the encoder's 128-dim pooled output is used for evaluation.

---

## Dataset Interface

The trainer imports your dataset class dynamically — no file is hard-coded. Your dataset must:

1. Accept `split="train" | "val"` in its constructor (plus any kwargs you configure).
2. Return dicts from `__getitem__` with:
   - `"context"` — tensor of shape `(C, T, H, W)` where `C=11, T=16, H=W=224`
   - `"target"` — tensor of the same shape
3. Apply all augmentations inside the dataset.

Point the config at your dataset module:

```yaml
dataset:
  module: "data"
  class_name: "ActiveMatterDataset"
  kwargs:
    num_frames: 16
    resolution: [224, 224]
    noise_std: 1.0
```

---

## Files

| File | Description |
|---|---|
| `model.py` | ConvEncoder, ConvPredictor, ResidualBlock, LayerNorm |
| `loss.py` | VICReg with shuffle-and-chunk |
| `scheduler.py` | Cosine LR with linear warmup |
| `train.py` | Training loop (DDP + AMP + gradient accumulation, W&B logging) |
| `eval_probe.py` | Linear probe + kNN regression on frozen encoder embeddings |
| `collapse_check.py` | Representation collapse diagnostics (effective rank, channel stats, kNN identity) |
| `config.yaml` | Training config (mirrors train_activematter_small.yaml) |
| `run_slurm_job.sbatch` | SLURM job script for NYU HPC |

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

Set `dataset.module` and `dataset.kwargs` to point at your `ActiveMatterDataset`. Also set:
- `out_path` — where checkpoints go
- `run_name` — wandb run name and subdir under `out_path`
- `wandb_project` — wandb project to log to

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
python train.py --config config.yaml train.lr=5e-4 train.num_epochs=50
```

### 6. Resume from checkpoint

```bash
python train.py --config config.yaml --resume ./checkpoints/<run>/latest.pt
```

### 7. Dry run (smoke test without wandb)

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

## Evaluation

The frozen encoder (128-dim global-average-pooled output) is evaluated by predicting the physical parameters α and ζ using:

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

`collapse_check.py` quantifies how well the encoder uses its feature space. Run it on any checkpoint:

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
- **Near-unit channel fraction** — channels with std ∈ [0.9, 1.1]; high values indicate the encoder is only barely satisfying the VICReg std hinge (partial collapse signal)
- **kNN identity rate** — whether distinct trajectories map to distinct features

## Memory & Throughput Notes

Even at only 3.3M parameters, this model consumes ~100 GB VRAM at batch size 8 because intermediate activations on 16×224×224×11 inputs are large. Knobs if you hit OOM:

- `train.batch_size` — drop from 8 to 4 or 2; grad accum compensates
- `train.amp_dtype: "bf16"` (default) — keeps activations in bfloat16
- `dataset.num_frames` — dropping from 16 to 8 halves the time dimension throughout
- `dataset.kwargs.resolution: [112, 112]` — halves H×W (4× activation reduction)
- Enable `torch.utils.checkpoint` in res blocks (not wired in here yet)

---

## What's Intentionally Not Here

- No masking beyond the temporal context→target split (Option A baseline).
- No EMA target encoder. VICReg's variance term prevents collapse without one — this is what distinguishes conv-JEPA from V-JEPA.
- No separate target projection head. Encoder output is used directly.

---

## Next Steps

1. Run linear probe + kNN evaluation on frozen encoder features against the `alpha`/`zeta` regression targets.
2. Check for representation collapse: monitor `std_loss` — if it stays near the hinge (~0 after warmup), variance is healthy.
3. Ablate stride (1 vs 4 vs 16) for a clean story on overlap effects.
4. Consider Option C (hybrid conv-stem + transformer) as a contribution experiment.
