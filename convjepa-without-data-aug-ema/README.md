# JEPA Baseline — Option A (conv-JEPA) for active_matter

A clean, self-contained implementation of the reference conv-JEPA baseline
described in the final project spec. Follows the architecture and training
recipe from Qu et al., "Representation Learning for Spatiotemporal Physical
Systems."

## What this trains

- **Encoder**: 5-stage ConvNeXt, dims `[16, 32, 64, 128, 128]`, residual
  blocks `[3, 3, 3, 9, 3]`. 3D convs with per-stage 2× downsampling until
  T collapses to 1, then 2D residual blocks. 2.47M parameters.
- **Predictor**: Small conv head (C → 2C → C). 0.80M parameters.
- **Loss**: VICReg on dense spatial embeddings (sim=2, std=40, cov=2).
- **Self-supervision**: Given 16 context frames, predict the latent of the
  next 16 target frames. No spatial/patch masking — the only "mask" is
  the temporal split.

Total: ~3.3M parameters, matching the reference baseline.

## Files

```
jepa_baseline/
├── model.py        # ConvEncoder, ConvPredictor, ResidualBlock, LayerNorm
├── loss.py         # VICReg with shuffle-and-chunk
├── scheduler.py    # Cosine LR with linear warmup
├── train.py        # Main training script (DDP + AMP + grad accum)
├── config.yaml     # Training config (mirrors train_activematter_small.yaml)
└── _sanity.py      # Shape/gradient check (run once after setup)
```

## Data interface

The trainer imports your dataset class dynamically — no file is hard-coded.
Your dataset must:

1. Have a constructor that accepts `split="train" | "val"` (plus any kwargs
   you configure).
2. Return dicts from `__getitem__` with:
   - `"context"` — tensor of shape `(C, T, H, W)` where `C=11, T=16,
     H=W=224` for the active_matter default.
   - `"target"` — tensor of the same shape.
3. Apply all augmentations (roll, noise, etc.) inside the dataset.

Point the config at your dataset module:

```yaml
dataset:
  module: "data"                   # Python module containing the class
  class_name: "ActiveMatterDataset"
  kwargs:
    num_frames: 16
    resolution: [224, 224]
    noise_std: 1.0
    # ... whatever else your dataset needs
```

## Quickstart

### 1. Verify the model builds

```bash
cd jepa_baseline
python _sanity.py
```

Expected output:
```
[PARAMS] encoder:   2,468,720
[PARAMS] predictor: 801,664
[PARAMS] total:     3,270,384
...
[GRAD ] backward OK
```

### 2. Edit the config

Set `dataset.module` and `dataset.kwargs` so the trainer can build your
`ActiveMatterDataset`. You'll probably also want to set:

- `out_path` — where checkpoints go
- `run_name` — becomes the wandb run name and subdir under `out_path`
- `wandb_project` — wandb project to log to

### 3. Single-GPU training

```bash
python train.py --config config.yaml
```

### 4. Multi-GPU training

```bash
torchrun --nproc_per_node=4 train.py --config config.yaml
```

The trainer auto-detects distributed mode from environment variables set by
torchrun. Gradient accumulation automatically adjusts so the effective
global batch reaches `train.target_global_batch_size` (default 256).

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

## Checkpoints

Per epoch (if `save_every` allows), the trainer saves:
- `epoch_N.pt` — full training state at epoch N
- `latest.pt` — always the most recent
- `best.pt` — lowest validation loss so far

Each checkpoint contains: `encoder`, `predictor`, `optimizer`, `scheduler`,
`epoch`, `global_step`, `best_val_loss`, and the full config for
reproducibility.

## Memory & throughput notes

The reference paper notes that even at only 3.3M params, this model
consumes ~100 GB VRAM at batch size 8 because the intermediate activations
on 16×224×224×11 inputs are enormous. Knobs if you hit OOM:

- `train.batch_size` — drop from 8 to 4 or 2; grad accum will compensate.
- `train.amp_dtype: "bf16"` (default) — keeps activations in bf16.
- `dataset.num_frames` — dropping from 16 to 8 halves time dim throughout.
- `dataset.kwargs.resolution: [112, 112]` — halves H×W (4× activation cut).
- Enable `torch.utils.checkpoint` in res blocks (not wired in here yet).

## What's intentionally *not* here

- No masking beyond the temporal context→target split. (Option A baseline.)
- No EMA target encoder. VICReg's variance term prevents collapse without
  one — this is what distinguishes conv-JEPA from V-JEPA.
- No target projector / separate projection head. Encoder output is used
  directly.

If you want to ablate any of these, see the architecture options menu from
our earlier discussion.

## Training recipe (from the paper, encoded in `config.yaml`)

| Hyperparameter            | Value          |
|---------------------------|----------------|
| Optimizer                 | AdamW, betas=(0.9, 0.95) |
| Learning rate             | 1e-3           |
| Weight decay              | 0.05           |
| LR schedule               | Cosine, 2-epoch warmup, min=1e-6 |
| Gradient clip             | 1.0            |
| Per-device batch size     | 8              |
| Target global batch size  | 256 (via grad accum) |
| Epochs                    | 30             |
| VICReg sim / std / cov    | 2 / 40 / 2     |
| VICReg chunks per batch   | 5              |
| Mixed precision           | bf16           |
| Noise augmentation        | std=1.0 (in dataset) |

## Next steps after this baseline trains

1. Run linear probe + kNN evaluation on frozen encoder features against
   the `alpha`/`zeta` regression targets.
2. Check for representation collapse: monitor `std_loss` — if it stays
   near the hinge (~0 after warmup), variance is healthy.
3. Ablate stride (1 vs 4 vs 16) for a clean story on overlap effects.
4. Then decide whether to try Option C (hybrid conv-stem + transformer)
   as your "contribution" experiment.
