# VideoMAE for Active Matter Physics Simulations

Self-supervised spatiotemporal representation learning using Masked Autoencoders (VideoMAE) on the `active_matter` physics simulation dataset. Trained from scratch — no pretrained weights.

---

## Overview

VideoMAE learns representations by masking 90% of spatiotemporal patches from a physics video and training a ViT encoder to reconstruct the masked content through a lightweight decoder. After training, the decoder is discarded and the frozen encoder is evaluated via linear probe and kNN regression on two continuous physical parameters (`alpha`, `zeta`).

The key insight from the [VideoMAE paper (Tong et al., 2022)](https://arxiv.org/abs/2203.12602) is that **temporal tube masking** — masking the same spatial positions across all time steps — forces the model to learn genuine spatiotemporal dynamics rather than simply interpolating nearby frames. This makes the 90% masking ratio tractable and highly effective for video data.

---

## Dataset: `active_matter`

**Source:** [polymathic-ai/active_matter](https://huggingface.co/datasets/polymathic-ai/active_matter) (HuggingFace), stored locally as HDF5 files.

### Splits
| Split | Files | Simulations | Samples (stride=1) |
|-------|-------|-------------|-------------------|
| Train | 45    | ~175 sims   | 11,550            |
| Valid | 16    | ~60 sims    | ~3,960            |
| Test  | —     | —           | ~4,290            |

### Physical Fields (11 channels)
Each simulation is a 2D active matter system (e.g., bacterial colonies, cytoskeletal networks) evolved over time. Each sample contains 5 physical tensor fields:

| Field | Key | Shape per frame | Channels | Description |
|-------|-----|-----------------|----------|-------------|
| Concentration | `t0_fields/concentration` | (256, 256) | 1 | Scalar density field |
| Velocity | `t1_fields/velocity` | (256, 256, 2) | 2 | 2D velocity vector field |
| Orientation tensor | `t2_fields/D` | (256, 256, 2, 2) | 4 | Nematic order tensor |
| Strain-rate tensor | `t2_fields/E` | (256, 256, 2, 2) | 4 | Symmetric strain-rate tensor |

These are stacked channel-wise → **11 channels** per frame.

### Raw Dimensions
```
HDF5 shape: (num_sims=3, num_timesteps=81, H=256, W=256)
```

### Prediction Targets (evaluation only)
- **`alpha`** — activity parameter (5 discrete values), controls energy injection by active units
- **`zeta`** — friction/alignment parameter (9 discrete values), controls dissipation

Labels are **never used during training** — only for linear probe / kNN evaluation.

---

## Data Pipeline

### Sliding Window Sampling
Each training sample is a 16-frame temporal window extracted via a stride-1 sliding window:
```
window = 16 frames
max_start = 81 - 16 = 65 per simulation
samples per simulation = 66  (stride=1)
```

### Preprocessing (per sample)
1. **Load** 16 consecutive frames from HDF5 → `(T=16, 11, 256, 256)`
2. **Z-score normalize** per channel across the full clip:
   ```
   mean = clip.mean(axis=(T, H, W))   # shape (11,)
   std  = clip.std(axis=(T, H, W))    # shape (11,)
   clip = (clip - mean) / (std + 1e-6)
   ```
3. **Random spatial crop** 224×224 from 256×256 (train) / center crop (val/test)
4. **Random flip + rotation** (horizontal flip, vertical flip, 90°/180°/270° rotation) — valid augmentations for the isotropic simulations
5. **Permute** `(T, C, H, W)` → `(C, T, H, W)` = `(11, 16, 224, 224)`

No Gaussian noise is applied — the 90% masking ratio serves as the primary stochastic augmentation.

---

## Model Architecture

### Patch Embedding (Tubelet Embedding)
Input video `(B, C=11, T=16, H=224, W=224)` is divided into non-overlapping 3D patches called **tubelets**:

```
Tubelet size:  (t=2, h=16, w=16)
Grid:          T/2 × H/16 × W/16  =  8 × 14 × 14  =  1,568 patches
```

Implemented as a single `Conv3D(in=11, out=192, kernel=(2,16,16), stride=(2,16,16))` followed by LayerNorm. This maps the full video to a sequence of 1,568 tokens in one pass.

### Positional Embedding
3D sinusoidal positional embeddings (non-learnable) are added to every token before masking. Each token's position is encoded independently along the temporal (t), height (h), and width (w) axes:

```
embed_dim = 192 = 64 (t) + 64 (h) + 64 (w)
```

### Temporal Tube Masking
**90% masking ratio** applied as temporal tubes — the same set of spatial `(h, w)` positions is masked across **all** 8 temporal positions:

```
Total patches:         1,568
Visible spatial:       int(196 × 0.10) = 19  (out of 14×14=196)
Visible spatiotemporal: 19 × 8  =  152  tokens seen by encoder
Masked tokens:         1,568 - 152  =  1,416
```

This tube masking strategy prevents trivial reconstruction by temporal interpolation, forcing the model to understand physical dynamics.

### Encoder (ViT-Tiny)
A standard Vision Transformer operating **only on the 152 visible tokens**:

| Hyperparameter | Value |
|----------------|-------|
| Architecture | ViT-Tiny |
| `embed_dim` | 192 |
| `depth` | 12 transformer blocks |
| `num_heads` | 3 |
| `mlp_ratio` | 4.0 |
| `dropout` | 0.0 |
| Input tokens | ~152 (10% of 1,568) |
| Output | `(B, 152, 192)` |

**Encoder parameters: 6,420,672 (~6.4M)**

Each transformer block: LayerNorm → Multi-Head Self-Attention → LayerNorm → MLP (GELU activation).

### Decoder (Lightweight, training only)
The decoder reconstructs **all 1,568 patches** from the 152 visible encoder tokens:

1. Project encoder output: `192 → 96` via linear layer
2. Fill masked positions with a **learned mask token** `(1, 1, 96)`
3. Add full 3D sinusoidal positional embedding (96-dim) to all 1,568 positions
4. Run 4 lightweight transformer blocks (dim=96, heads=3)
5. Linear prediction head: `96 → patch_dim` where `patch_dim = 2 × 16 × 16 × 11 = 5,632`

| Hyperparameter | Value |
|----------------|-------|
| `dec_embed_dim` | 96 |
| `dec_depth` | 4 |
| `dec_heads` | 3 |
| Output | `(B, 1568, 5632)` |

**Decoder parameters: 1,012,480 (~1.0M)** — discarded after training.

### Parameter Budget
| Component | Parameters |
|-----------|-----------|
| Encoder | 6,420,672 |
| Decoder | 1,012,480 |
| **Total** | **7,433,152** |

Well within the 100M parameter limit.

---

## Training Objective

### Reconstruction Loss
MSE over **masked patches only**, with per-patch normalization of the target:

```python
# 1. Extract ground truth patches from input video
target = patchify(x)           # (B, 1568, 5632)

# 2. Per-patch normalization (removes local mean/variance bias)
mean   = target.mean(dim=-1, keepdim=True)
var    = target.var(dim=-1,  keepdim=True)
target = (target - mean) / (var + 1e-6).sqrt()

# 3. MSE on masked patches only
loss = ((pred - target) ** 2).mean(dim=-1)   # (B, 1568)
loss = (loss * mask).sum() / mask.sum()      # scalar
```

Per-patch normalization is the standard approach from the original MAE paper — it prevents the model from focusing on global intensity differences rather than local structure.

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 200 |
| Batch size | 8 (per GPU) |
| Effective batch | 64 (grad accumulation × 8) |
| Optimizer | AdamW (`β1=0.9`, `β2=0.95`) |
| Base LR | 1.5×10⁻⁴ |
| LR schedule | Cosine decay with 20-epoch warmup |
| Weight decay | 0.05 (weights only, not biases/LN) |
| Gradient clip | 1.0 |
| Mixed precision | bf16 (A100) |
| Hardware | 1× NVIDIA A100 40GB |

### Fault Tolerance
- **Step-level checkpoints** every 50 optimizer steps (`latest.pt`)
- **Epoch-level checkpoints** every 5 epochs
- **Best model** saved whenever val loss improves (`best.pt`)
- `SIGUSR1` / `SIGTERM` handler saves checkpoint immediately on preemption
- `#SBATCH --requeue` automatically re-submits the job; training resumes from `latest.pt`
- W&B run ID persisted to `wandb_run_id.txt` so logging continues seamlessly across preemptions

---

## Evaluation

After training, the **encoder is frozen** and representations are extracted for all splits (stride=2 for speed). Two methods are evaluated:

### 1. Linear Probe (Ridge Regression)
A single Ridge regression layer fit on frozen embeddings. Targets are z-score normalized before fitting.

### 2. kNN Regression (k=20)
k-Nearest Neighbours regression (k=20, Euclidean distance) on standardized embeddings. No model parameters are trained.

Both methods predict `alpha` and `zeta` as continuous values. Metric: **MSE on z-score normalized targets** (random baseline ≈ 1.0).

### Early Results (checkpoint from early training)
| Method | alpha (val) | zeta (val) | alpha (test) | zeta (test) |
|--------|-------------|------------|--------------|-------------|
| Linear Probe | **0.187** | 0.224 | **0.350** | 0.221 |
| kNN (k=20)   | 0.623 | **0.166** | 0.574 | **0.195** |

All results well below the random baseline of 1.0. Results expected to improve with continued training.

---

## File Structure

```
video MAe/
├── dataset.py      — VideoMAEDataset (train/val/test) + VideoMAEEval (embedding extraction)
├── model.py        — PatchEmbed3D, tube masking, ViTEncoder, MAEDecoder, VideoMAE
├── train.py        — Training loop with MAE objective, checkpointing, W&B logging
├── evaluate.py     — Linear probe + kNN evaluation on frozen encoder embeddings
├── submit.slurm    — SLURM training job (A100, 16h, --requeue)
└── eval.slurm      — SLURM evaluation job (T4, 2h)
```

---

## Usage

### Training
```bash
# Fresh start
python train.py

# Resume from checkpoint (auto-detected by submit.slurm)
python train.py --resume /scratch/sb10583/checkpoints/videomae-v1/latest.pt

# Dry run (1 epoch)
python train.py --dry-run
```

### Evaluation
```bash
python evaluate.py \
  --checkpoint /scratch/sb10583/checkpoints/videomae-v1/best.pt \
  --data-dir   /scratch/sb10583/data/data \
  --knn-k      20 \
  --stride     2
```

### SLURM (NYU HPC)
```bash
sbatch "video MAe/submit.slurm"   # training
sbatch "video MAe/eval.slurm"     # evaluation
```

---

## References

- Tong, Z., Song, Y., Wang, J., & Wang, L. (2022). **VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training.** NeurIPS 2022. [arXiv:2203.12602](https://arxiv.org/abs/2203.12602)
- He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). **Masked Autoencoders Are Scalable Vision Learners.** CVPR 2022. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
- Polymathic AI. **active_matter dataset.** [HuggingFace](https://huggingface.co/datasets/polymathic-ai/active_matter)
