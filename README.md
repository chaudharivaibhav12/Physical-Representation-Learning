# Self-Supervised Representation Learning for Active Matter Simulations

**CSCI-GA 2572 Deep Learning — Spring 2026, NYU Courant Institute**

**Team:** Sarvesh Bodke (sb10583) · Vaibhav Chaudhari (vc2836) · Ojaswi Kaushik (ok2287)

---

## Overview

We study self-supervised representation learning on the `active_matter` physical simulation dataset. Five families of architectures are trained from scratch and evaluated by how well frozen representations predict the underlying physical parameters **α** (active dipole strength) and **ζ** (steric alignment) via linear probing and kNN regression.

The key insight motivating our approach: a physical simulation `(T × H × W × 11)` is structurally analogous to a video `(T × H × W × 3)`, allowing direct adaptation of state-of-the-art video SSL architectures.

---

## Dataset

**`active_matter`** from [The Well](https://github.com/PolymathicAI/the_well) (`polymathic-ai/active_matter`, ~52 GB on HuggingFace).

Each simulation is a spatiotemporal field of 11 physical channels (concentration, velocity, orientation tensor, strain-rate tensor) at 256×256 resolution over 81 time steps.

| Split | Files | Samples |
|-------|-------|---------|
| Train | 45 | 8,750 |
| Validation | 16 | 1,200 |
| Test | 21 | 1,300 |

Labels (α, ζ) are withheld during pre-training and used only for downstream evaluation.

```bash
huggingface-cli download polymathic-ai/active_matter --repo-type dataset --local-dir ./data
```

---

## Repository Structure

```
Physical-Representation-Learning/
│
├── convjepa-with-data-aug-vicreg/   # Conv-JEPA baseline (Vaibhav) — with augmentation
├── convjepa-without-data-aug-vicreg/ # Conv-JEPA — no augmentation, VICReg
├── convjepa-without-data-aug-ema/   # Conv-JEPA — no augmentation, EMA target
│
├── ViT-JEPA-v2/                     # ViT-JEPA with transformer predictor (Vaibhav)
├── VIT-JEPA-OJASWI-patch-32/        # ViT-JEPA patch-32 ablation (Ojaswi)
├── VIT-JEPA-OJASWI-patch-16/        # ViT-JEPA patch-16 ablation (Ojaswi)
├── ViT-JEPA-EMA/                    # ViT-JEPA with EMA + block masking (Ojaswi)
│
├── vision-transformer-v1/           # VICReg ViT v1 — 20 epochs, stride=4 (Sarvesh)
├── vision-transformer-v2/           # VICReg ViT v2 — 100 epochs, stride=1 (Sarvesh)
├── ViT Jepa sarvesh/                # ViT-JEPA with EMA + block masking (Sarvesh)
├── video MAe/                       # VideoMAE masked autoencoding (Sarvesh)
│
├── final_report.md                  # Full paper writeup
└── Final_Project.md                 # Project specification
```

Each model folder contains: `model.py`, `dataset.py`, `train.py`, `evaluate.py`, `submit.slurm`, `eval.slurm`, and eval results (`eval_best.json`, `eval_best.txt`).

---

## Models

### 1. Conv-JEPA (Vaibhav)
ConvNeXt encoder (5-stage, ~3.3M params) trained with a predictive coding objective. Three variants ablate data augmentation and collapse-prevention strategy (VICReg variance term vs EMA target encoder).

### 2. VICReg with 3D Vision Transformer (Sarvesh)
3D ViT encoder (patch 32×32, embed_dim=384, 6L) with a 3-layer MLP projector trained under VICReg loss on two temporally offset views. Two variants ablate training scale (20 vs 100 epochs, stride 4 vs 1).

### 3. ViT-JEPA with Temporal Predictive Coding (Vaibhav + Ojaswi)
ViT encoder + shallow transformer predictor that maps context token sequences to predicted target embeddings, with VICReg loss. Ojaswi's variants ablate patch size (32 vs 16).

### 4. ViT-JEPA with EMA + Block Masking (Ojaswi + Sarvesh)
EMA target encoder with cosine-annealed momentum (0.996→0.9999) and spatiotemporal block masking. MSE loss between L2-normalized predictor output and target encoder output. No VICReg needed.

### 5. VideoMAE (Sarvesh)
ViT-Tiny encoder (12L, dim=192) trained with 90% temporal tube masking and reconstruction loss. Decoder discarded at eval.

---

## Evaluation

All models are evaluated with the encoder **fully frozen**:

1. **Linear Probe** — Ridge regression (`sklearn`, alpha=1.0) on frozen embeddings
2. **kNN Regression** — k=20, cosine similarity

Targets (α, ζ) are z-score normalized. MSE reported on val and test sets. Random baseline ≈ 1.0.

### Results (Test Set, best.pt)

| Model | Params | LP MSE α | LP MSE ζ | kNN MSE α | kNN MSE ζ | Embed std |
|-------|--------|----------|----------|-----------|-----------|-----------|
| VICReg ViT v1 (20ep) | 23.6M | 0.553 | 0.744 | 0.704 | 0.961 | 0.091 ⚠️ |
| **VICReg ViT v2 (100ep)** | 23.6M | **0.063** | **0.219** | **0.085** | **0.286** | 0.366 ✓ |
| ViT-JEPA sarvesh | 6.85M | 0.101 | 0.282 | 0.317 | 0.337 | 0.314 ✓ |
| VideoMAE | 6.4M | 0.094 | 0.173 | 0.272 | 0.237 | 0.100 ✓ |

Lower is better. Random baseline ≈ 1.0.

---

## Training on NYU HPC

```bash
# Submit training job
sbatch vision-transformer-v2/submit.slurm

# Submit eval job
sbatch vision-transformer-v2/eval.slurm

# Monitor
squeue -u $USER
tail -f /scratch/$USER/logs/<job_name>_<jobid>.log
```

All jobs use:
- Partition: `c12m85-a100-1` (A100) for training, `n1s8-t4-1` (T4) for eval
- `#SBATCH --requeue` for automatic restart on preemption
- Step-level checkpointing every 20 steps (`latest.pt`) + best checkpoint (`best.pt`)
- W&B logging (project: `DL`, entity: `sb10583-`)

---

## References

1. McCabe et al. "The Well." NeurIPS 2024.
2. Qu et al. "Representation Learning for Spatiotemporal Physical Systems." arXiv:2603.13227.
3. Assran et al. "I-JEPA." CVPR 2023.
4. He et al. "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022.
5. Tong et al. "VideoMAE." NeurIPS 2022.
6. Bardes et al. "VICReg." ICLR 2022.
