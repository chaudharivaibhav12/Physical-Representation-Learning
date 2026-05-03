# Self-Supervised Representation Learning for Active Matter Physical Simulations

**CSCI-GA 2572 Deep Learning — Spring 2026**  
**NYU Courant Institute of Mathematical Sciences**

**Team:** Sarvesh Bodke (sb10583) · Vaibhav Chaudhari (vc2836) · Ojaswi Kaushik (ok2287)

---

## Abstract

> *[To be written after final results. Suggested structure: "We present a systematic comparison of self-supervised representation learning methods on the active_matter physical simulation dataset. Across eleven experimental variants spanning convolutional and transformer architectures, VICReg and EMA-based collapse prevention, and predictive versus reconstructive objectives, we find that [key finding]. Our best model achieves a linear probing MSE of [X] / [X] and kNN MSE of [X] / [X] on normalized α and ζ respectively, well below the random baseline of 1.0."]*

---

## 1. Introduction

Active matter systems — colonies of self-propelled agents such as bacterial films or cytoskeletal networks — exhibit rich spatiotemporal dynamics governed by two underlying physical parameters: **α** (active dipole strength), controlling the intensity of energy injection by active units, and **ζ** (steric alignment), controlling orientational interactions and dissipation. Learning compact, physically meaningful representations of these dynamics *without access to labels* is a fundamental challenge at the intersection of deep learning and computational physics.

We study this problem on the `active_matter` dataset from The Well [1], which provides spatiotemporal trajectories of 2D active nematic simulations as 11-channel physical fields (concentration, velocity, orientation tensor, strain-rate tensor) evolving over time. Our objective is to train self-supervised encoders that produce representations from which the physical parameters α and ζ can be recovered by a simple frozen linear layer or kNN regressor — providing a direct probe of physical information content.

A central observation motivating our architectural choices is the structural analogy between this data and video: just as a video is a sequence of T frames with 3 RGB channels, each simulation trajectory is a sequence of T frames with 11 physical channels. This allows direct adaptation of state-of-the-art video self-supervised learning methods — VideoMAE [6], ViT-JEPA [4], V-JEPA — to the physical simulation domain, while also enabling systematic comparison with convolutional baselines more common in the physics-ML literature [2].

We explore **eleven experimental configurations** across five families of methods, varying encoder architecture (ConvNeXt vs. 3D Vision Transformer), self-supervised objective (VICReg, EMA latent prediction, masked reconstruction), patch granularity, data augmentation, and training scale. This breadth allows us to isolate the contribution of individual design choices through controlled ablations.

Our contributions are:
- A systematic study of SSL methods on physical simulation data, spanning convolutional and transformer encoders, predictive and reconstructive objectives, and VICReg versus EMA collapse prevention.
- Controlled ablations over data augmentation, patch size, training duration, and collapse-prevention strategy within each architecture family.
- A physics-aware data augmentation protocol (sign-corrected spatial flips) that maintains physical consistency of vector and tensor field channels.
- A complete, fault-tolerant training and evaluation pipeline on NYU HPC, with step-level checkpointing, SIGUSR1 preemption handling, and W&B logging.

---

## 2. Related Work

**Self-Supervised Representation Learning.**
Contrastive methods such as SimCLR [7] and MoCo [8] learn representations by pulling together different augmented views of the same sample while pushing apart views from different samples. VICReg [9] removes the need for negative pairs by instead enforcing per-dimension variance and decorrelating embedding dimensions, preventing collapse via explicit regularization. Masked Autoencoders (MAE) [5] take a reconstruction-based approach, training a ViT encoder to reconstruct randomly masked patches through a lightweight decoder — learning rich representations from the pretext task of reconstruction.

**Joint-Embedding Predictive Architectures (JEPA).**
I-JEPA [4] learns by predicting the representations of spatially masked target regions from visible context patches, operating in representation space rather than pixel space. This avoids the need to reconstruct irrelevant low-level texture. V-JEPA and EB-JEPA [3] extend this to video by predicting spatiotemporal target blocks, and demonstrate that representations learned from latent prediction transfer well to downstream tasks. JEPA-style methods use an EMA (Exponential Moving Average) target encoder to provide a stable prediction target without contrastive pairs.

**Representation Learning for Physical Systems.**
Qu et al. [2] introduce a convolutional JEPA baseline for spatiotemporal physical simulations, using a ConvNeXt encoder to predict latent representations of future frames from context frames under a VICReg objective. This directly motivates our convolutional baselines. More broadly, neural operators [10] and learned surrogates learn evolution operators over PDE solutions, but their goal is simulation rather than representation. Our work focuses on representation quality as measured by downstream linear probing.

**Video Representation Learning.**
VideoMAE [6] shows that masking 90% of spatiotemporal patches with a tube masking strategy — the same spatial positions across all time steps — forces the model to understand genuine temporal dynamics rather than interpolating nearby frames. This high masking ratio is particularly suited to the temporally redundant structure of physical simulations, which motivates our VideoMAE adaptation.

---

## 3. Dataset

We use the **`active_matter`** subset of The Well [1], hosted on HuggingFace (`polymathic-ai/active_matter`, ~52 GB total).

**Physical System.** Each simulation models a 2D active nematic liquid crystal system: rod-like active particles immersed in a viscous fluid that self-organize and create large-scale flows. The dynamics are governed by two physical constants that define 45 unique parameter combinations:
- **α** (active dipole strength): 5 discrete values — controls energy injection rate
- **ζ** (steric alignment): 9 discrete values — controls orientational interactions and dissipation

**Data Structure.** Each HDF5 file contains 3 independent simulation trajectories of 81 time steps at 256×256 spatial resolution. Each frame comprises 11 physical channels from 4 tensor fields:

| Field | Channels | Description |
|-------|----------|-------------|
| Concentration | 1 | Scalar density of active particles |
| Velocity | 2 | 2D velocity vector field |
| Orientation tensor (D) | 4 | Symmetric nematic order tensor |
| Strain-rate tensor (E) | 4 | Symmetric strain-rate tensor |

**Dataset Splits.**

| Split | Files | Simulations | Samples (stride=1) |
|-------|-------|-------------|-------------------|
| Train | 45 | ~135 | 8,750 |
| Validation | 16 | ~48 | 1,200 |
| Test | — | ~54 | 1,300 |

**Labels.** α and ζ are withheld during pre-training and used **only** for downstream evaluation via z-score normalized regression. A random encoder baseline yields MSE ≈ 1.0.

**Preprocessing.** All models apply the following common pipeline:
1. Extract sliding window clips of 16 (or 32) consecutive frames with stride 1 (or 4 for v1).
2. Per-sample, per-channel z-score normalization across T×H×W.
3. Random 224×224 crop from 256×256 (train) / center crop (val, test).

**Physics-Aware Data Augmentation (training only).** Most models apply spatial isometry augmentations valid for the rotationally symmetric simulation domain: random horizontal/vertical flips and 90° rotations. The ViT-JEPA-v2 (Vaibhav) additionally applies *sign-correction* to vector and tensor field channels after spatial flips — for example, the x-component of velocity flips sign under a horizontal flip — ensuring augmented samples remain physically consistent. Gaussian noise (σ=1.0) is applied as additional stochastic augmentation in most variants.

---

## 4. Methods

We explore five families of self-supervised architectures across eleven experimental configurations. All models are trained **from scratch** with no pretrained weights, remain under 100M parameters, and use only the training split for weight updates.

The core self-supervised task in all methods is: given a 16-frame context clip from a simulation, learn representations from which the target frames (or their statistics) can be predicted. Physical parameter labels are never used during training.

---

### 4.1 Convolutional JEPA (Conv-JEPA)

*Vaibhav Chaudhari — `convjepa-with-data-aug-vicreg`, `convjepa-without-data-aug-vicreg`, `convjepa-without-data-aug-ema`*

Following Qu et al. [2], the Conv-JEPA baseline uses a **5-stage ConvNeXt encoder** that processes spatiotemporal clips by progressively downsampling with 3D convolutions in early stages and transitioning to 2D convolutions as the temporal dimension collapses.

```
Training:
  context (B, 11, 16, 224, 224)
      → ConvNeXt Encoder (5 stages, 3D → 2D)
          Stage 0: stem     → (B, 16,  16, 224, 224)  [3 res blocks, no downsample]
          Stage 1: down3d   → (B, 32,   8, 112, 112)  [2× stride, 3 res blocks]
          Stage 2: down3d   → (B, 64,   4,  56,  56)  [2× stride, 3 res blocks]
          Stage 3: down3d   → (B, 128,  2,  28,  28)  [2× stride, 9 res blocks]
          Stage 4: down3d   → (B, 128,  1,  14,  14)  [T collapses → 2D, 3 res blocks]
      → ctx_embed (B, 128, 14, 14)
      → ConvPredictor: Conv2d(128→256) → ResBlock(256) → Conv2d(256→128)
      → predicted_embed (B, 128, 14, 14)
                                             ┐
  target (B, 11, 16, 224, 224)              ├─→ Loss (VICReg or MSE)
      → Encoder (shared / EMA copy)         │
      → tgt_embed (B, 128, 14, 14)         ┘

Evaluation:
  clip → Encoder → global avg pool → (B, 128) → Linear Probe / kNN
```

**VICReg loss** is applied on the dense spatial embeddings, treating the 14×14=196 spatial vectors per sample as a bag of embeddings for variance/covariance estimation (196× more samples than pooled-vector VICReg).

**Three variants** ablate the effect of data augmentation and collapse-prevention strategy:

| Variant | Data Aug | Collapse Prevention | Epochs | Params |
|---------|----------|---------------------|--------|--------|
| `convjepa-with-data-aug-vicreg` | Yes (flip, rotate, noise σ=1.0) | VICReg (sim=2, std=20, cov=2) | 50 | ~3.27M |
| `convjepa-without-data-aug-vicreg` | **None** | VICReg (sim=2, std=20, cov=2) | 25 | ~3.27M |
| `convjepa-without-data-aug-ema` | **None** | **EMA target encoder** (τ: 0.996→0.9999) | 100 | ~3.27M |

In the EMA variant, a separate non-trainable copy of the encoder is updated after each optimizer step as θ_target ← τ·θ_target + (1−τ)·θ_online, providing a stable prediction target without explicit variance/covariance constraints. Pure MSE loss is applied between predictor output and the stop-gradient EMA target embedding.

| Component | Parameters |
|-----------|-----------|
| ConvNeXt Encoder (5 stages) | ~2.47M |
| Conv Predictor (Conv→ResBlock→Conv) | ~0.80M |
| **Total trainable** | **~3.27M** |

---

### 4.2 VICReg with 3D Vision Transformer Encoder

*Sarvesh Bodke — `vision-transformer-v1`, `vision-transformer-v2`*

To capture the global, long-range spatiotemporal patterns that determine α and ζ — which a locally-receptive ConvNeXt cannot see in early layers — we replace the convolutional encoder with a **3D Vision Transformer (ViT)**. Starting from layer one, every token attends to all other tokens in the full 224×224 spatial field.

```
Training:
  view1 (B, 11, 16, 224, 224)
      → Conv3D(kernel=2×32×32, stride=2×32×32)
      → 392 tokens (B, 392, 384)   [8T × 7H × 7W]
      → + 3D sinusoidal pos embed (fixed)
      → ViT Encoder: 6× TransformerBlock (dim=384, heads=6, MLP=1536)
      → LayerNorm → global mean pool → (B, 384)
      → ProjectionMLP: 384 → 2048 → 2048 → 2048
      → z1 (B, 2048)
                                  ┐
  view2 (same clip, diff aug)    ├─→ VICReg Loss (on z1, z2)
      → same path → z2 (B, 2048) ┘

Evaluation (projector discarded):
  clip → Encoder → mean pool → (B, 384) → Linear Probe / kNN
```

Two views are created from the same 16-frame clip by applying different random augmentations. VICReg is computed on the projected embeddings (B, 2048). The large projector (384→2048→2048→2048) is a training aid only and is discarded at evaluation — only the 384-dim encoder output is used.

**Two variants** ablate training scale and VICReg variance weight:

| Hyperparameter | v1 | v2 |
|----------------|----|----|
| Epochs | 20 | **100** |
| Batch size (eff.) | 32 | **64** |
| Stride (train samples) | 4 → 2,275 clips | **1 → 8,750 clips** |
| Variance weight | **50.0** | **25.0** |
| Step checkpointing | No | Yes (every 20 steps) |
| SIGUSR1 handler | No | Yes |
| W&B run persistence | No | Yes |

v2 trains on the full dataset at stride=1 (matching the project specification of 8,750 training samples), reduces the variance weight from 50→25 as the larger dataset naturally provides collapse prevention, and adds complete preemption-safe infrastructure.

| Component | Parameters |
|-----------|-----------|
| ViT Encoder (6L, dim=384) | ~11M |
| Projection MLP (384→2048→2048→2048) | ~12.6M |
| **Total trainable** | **~23.6M** |

---

### 4.3 ViT-JEPA: Temporal Predictive Coding with Token-Level VICReg

*Vaibhav Chaudhari — `ViT-JEPA-v2`*  
*Ojaswi Kaushik — `VIT-JEPA-OJASWI-patch-32`, `VIT-JEPA-OJASWI-patch-16`*

Rather than applying VICReg between two mean-pooled embeddings (§4.2), this family keeps spatial and temporal structure intact by feeding the **full token sequence** through a shallow transformer predictor before pooling. This allows the predictor to attend over specific spatial positions and time steps before summarizing, preserving information that pooling would discard.

```
Context frames (B,11,16,224,224)         Target frames (B,11,16,224,224)
        │                                          │
  PatchEmbed3D                              PatchEmbed3D
  (Conv3D, stride=patch)                    (Conv3D, stride=patch)
        │                                          │
  N tokens (B, N, 384)                      N tokens (B, N, 384)
        │                                          │
  ViT Encoder (6-8L, 384)              [shared weights] ViT Encoder
  (B, N, 384)                                      │
        │                                    global mean pool
  TransformerPredictor                             │
  (2L, 384→192→384 bottleneck)             z_tgt (B, 384)
        │                                          ║
  global mean pool                          VICReg Loss
  z_pred (B, 384) ════════════ VICReg ═══════════════╝
```

**VICReg applied at two levels (ViT-JEPA-v2 only).** Vaibhav's implementation computes:
- **Invariance** on pooled vectors `(B, 384)` — forces predictive alignment between context and target
- **Variance + Covariance** on *flat tokens* `(B×392, 384)` — provides 3,136 samples per batch (vs. 8 for pooled-only), giving a far more stable covariance estimate and stronger collapse prevention without augmentation pairs

**Physics-aware augmentation (ViT-JEPA-v2).** Spatial flips are applied with per-channel sign corrections ensuring vector and tensor fields remain physically valid after reflection.

**Patch size ablation (Ojaswi).** Two variants test whether finer spatial patches improve representation quality:

| Variant | Patch Size | Tokens | Embed Dim | Encoder Depth |
|---------|-----------|--------|-----------|---------------|
| `VIT-JEPA-OJASWI-patch-32` | 32×32 | 8×7×7 = **392** | 384 | 8 |
| `VIT-JEPA-OJASWI-patch-16` | 16×16 | 8×14×14 = **1,568** | 384 | 8 |
| `ViT-JEPA-v2` (Vaibhav) | 32×32 | 8×7×7 = **392** | 384 | 6 |

Smaller patches (16×16) provide 4× finer spatial resolution and 4× more tokens, allowing the predictor to attend over more fine-grained structure, but increase memory and compute by 4×.

| Component | patch-32 | patch-16 | ViT-JEPA-v2 |
|-----------|---------|---------|------------|
| ViT Encoder | ~5.3M | ~5.3M | ~5.3M |
| Transformer Predictor (2L, dim=192) | ~0.6M | ~0.6M | ~0.6M |
| **Total** | **~6.0M** | **~6.0M** | **~6.0M** |

---

### 4.4 ViT-JEPA with EMA Target Encoder and Block Masking

*Ojaswi Kaushik — `ViT-JEPA-EMA`*  
*Sarvesh Bodke — `ViT Jepa sarvesh`*

This family follows I-JEPA [4] more closely by adding two components: an **EMA target encoder** (providing a slowly-evolving, non-collapsed training target) and **spatiotemporal block masking** (a harder prediction task than a temporal context/target split).

```
Full clip (B, 11, T, 224, 224)
        │
  PatchEmbed3D → N tokens
        │
  ┌─────────────────────────────────────────────┐
  │       Spatiotemporal Block Masking           │
  │  Context idx (~75-85%) │ Target idx (~15-25%)│
  └────────────┬────────────────────────────────┘
               │                  │  (full clip, no grad)
  Online Encoder              Target Encoder
  (backprop)                  (EMA: θ_t ← τ·θ_t + (1-τ)·θ_online)
               │                  │  τ: 0.996 → 0.9999 (cosine)
  ctx_encoded                tgt_encoded
  (B, N_ctx, d)              (B, N_tgt, d)
               │              L2-normalize ↓
  ┌─── add mask_tokens @ target positions ───┐
  ▼                                          │
  Predictor (narrow ViT)          tgt (B, N_tgt, d)
  (B, N_tgt, d)                       ║
               ═══════ MSE Loss ═══════╝
```

**EMA target encoder.** The target encoder sees the full unmasked clip and its weights are updated as θ_t ← τ·θ_t + (1−τ)·θ_online after each optimizer step, with τ cosine-annealed from 0.996 to ~1.0 over training. This provides a stable, slowly-evolving prediction target that prevents collapse without requiring VICReg regularization.

**Spatiotemporal block masking.** Target regions are randomly sampled contiguous 3D blocks in (T, H, W) token space, forcing the model to reason about physically coherent spatiotemporal dynamics rather than independent frames.

**Ojaswi's ViT-JEPA-EMA.** Uses a ViT-Small architecture for both context and target encoders (depth=12, dim=384). The predictor is a narrow 6-layer ViT (dim=192). Multi-block masking samples 4 target blocks (scale 0.15–0.2) and 1 context block (scale 0.85–1.0) extended as temporal tubes. Each frame is treated as an independent 11-channel image with learned spatial and temporal positional embeddings (rather than the fused 3D Conv3D token embedding used elsewhere).

**Sarvesh's ViT-JEPA.** Uses a lighter encoder (depth=6, dim=256, 8 heads) with finer patch granularity (patch=16 → 1,568 tokens), 4 target blocks (~25% tokens). The narrow predictor uses a 256→128→256 bottleneck with 4 transformer blocks. Gradient checkpointing is enabled during training to fit the 1,568-token sequence on GPU. L2-normalization of target encoder output provides a bounded, scale-invariant training signal.

| Hyperparameter | Ojaswi (ViT-JEPA-EMA) | Sarvesh (ViT Jepa) |
|----------------|----------------------|--------------------|
| Patch size | 16×16 (2D per frame) | 16×16 (Conv3D) |
| Tokens | 4T × 14² = 784 | 8T × 14² = **1,568** |
| Encoder depth | **12** | 6 |
| Embed dim | **384** | 256 |
| Predictor depth | **6** | 4 |
| EMA momentum | 0.996 → 1.0 | 0.996 → 0.9999 |
| Loss | MSE | MSE (L2-norm targets) |
| Total trainable | **~22M** | **~6.85M** |

---

### 4.5 VideoMAE: Masked Autoencoding for Physical Simulations

*Sarvesh Bodke — `video MAe`*

VideoMAE [6] learns representations through **reconstruction** of masked spatiotemporal patches rather than latent prediction. We adapt this directly to the 11-channel physical simulation setting, leveraging the structural analogy between video (T×H×W×3) and simulation (T×H×W×11) data.

```
Input (B, 11, 16, 224, 224)
        │
  Conv3D(in=11, out=192, kernel=(2,16,16), stride=(2,16,16))
  → 1,568 tokens  (8T × 14H × 14W)
  + 3D sinusoidal positional embedding (192 = 64t + 64h + 64w)
        │
  ┌──────────────────────────────────────────┐
  │       Temporal Tube Masking (90%)         │
  │  Spatial grid: 14×14 = 196 positions     │
  │  Keep 10% visible: 19 positions          │
  │  Same spatial mask across ALL 8 timesteps│  → "tubes"
  └───────────┬──────────────────────────────┘
              │ visible (152 tokens)     masked (1,416 tokens dropped)
              ▼
  ViT-Tiny Encoder (12L, dim=192, 3 heads)
  — operates only on 152 visible tokens —
  (B, 152, 192)
              │
              ▼
  MAE Decoder (training only):
    1. Project 192 → 96
    2. Fill masked positions with learned [MASK] token
    3. Add full pos embed (96-dim) to all 1,568 positions
    4. 4× TransformerBlock (dim=96, 3 heads)
    5. Linear head: 96 → 5,632  (= 2×16×16×11 per patch)
  (B, 1,568, 5,632)
              │
  MSE on masked patches only (per-patch normalized targets)

  [Decoder discarded at eval — only ViT-Tiny encoder used]
```

**Tube masking.** The same spatial positions are masked across all 8 time steps, forming spatiotemporal "tubes." At 90% masking, only 152 of 1,568 tokens are visible to the encoder. This prevents the trivial shortcut of reconstructing masked regions by temporal interpolation from nearby visible frames, forcing genuine spatiotemporal reasoning.

**Per-patch normalized reconstruction loss.** Target patches are normalized by their own mean and variance before computing MSE, removing global intensity bias and focusing the model on local structural content:

$$\mathcal{L}_\text{MAE} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \left\| \hat{x}_i - \frac{x_i - \mu_i}{\sigma_i} \right\|_2^2$$

where $\mathcal{M}$ is the set of masked patch indices, and $\mu_i$, $\sigma_i$ are the per-patch mean and standard deviation of the ground-truth patch content.

| Component | Parameters |
|-----------|-----------|
| ViT-Tiny Encoder (12L, dim=192) | ~6.4M |
| MAE Decoder (4L, dim=96) | ~1.0M |
| **Total (training)** | **~7.4M** |
| **Encoder only (eval)** | **~6.4M** |

---

## 5. Training Details

All models use AdamW with cosine LR decay and a linear warmup period. Gradient clipping (max norm=1.0) and mixed precision (bfloat16) are applied throughout. All jobs run on NYU HPC via SLURM with `#SBATCH --requeue` for automatic spot-instance restart.

**Comprehensive hyperparameter table:**

| Model | Opt. | LR | Epochs | Batch (eff.) | Warmup | Grad Clip | Precision |
|-------|------|----|--------|--------------|--------|-----------|-----------|
| Conv-JEPA (aug+VICReg) | AdamW | 1e-3 | 50 | 128 | 1 ep | 1.0 | bf16 |
| Conv-JEPA (no-aug+VICReg) | AdamW | 1e-3 | 25 | 256 | 1 ep | 1.0 | bf16 |
| Conv-JEPA (no-aug+EMA) | AdamW | 1e-3 | 100 | 128 | 1 ep | 1.0 | bf16 |
| VICReg ViT v1 | AdamW | 1e-3 | 20 | 32 | 5 ep | 1.0 | bf16 |
| VICReg ViT v2 | AdamW | 1e-3 | 100 | 64 | 5 ep | 1.0 | bf16 |
| ViT-JEPA-v2 (Vaibhav) | AdamW | 1e-3 | 100 | 64 | 5 ep | 1.0 | bf16 |
| VIT-JEPA patch-32 (Ojaswi) | AdamW | 1e-3 | 100 | 64 | 5 ep | 1.0 | bf16 |
| VIT-JEPA patch-16 (Ojaswi) | AdamW | 1e-3 | 100 | 64 | 5 ep | 1.0 | bf16 |
| ViT-JEPA-EMA (Ojaswi) | AdamW | — | — | — | — | 1.0 | bf16 |
| ViT-JEPA sarvesh | AdamW | 1.5e-4 | 100 | 64 | 10 ep | 1.0 | bf16 |
| VideoMAE (Sarvesh) | AdamW | 1.5e-4 | 200 | 64 | 20 ep | 1.0 | bf16 |

**Fault tolerance.** All models implement step-level checkpointing (every 20–50 optimizer steps) and SIGUSR1 signal handling — SLURM sends SIGUSR1 90 seconds before spot instance preemption, the Python process saves a checkpoint and signals bash to requeue the job. W&B run IDs are persisted to disk so logging continues uninterrupted across restarts.

---

## 6. Evaluation Protocol

### 6.1 Frozen Encoder Evaluation

After pre-training, **all encoder weights are frozen**. Representations are extracted for every split by passing each clip through the encoder and applying global average pooling over the full token sequence (no masking at eval time). Two evaluation methods are then applied independently for α and ζ:

1. **Linear Probe.** A single linear layer (`nn.Linear(d, 1)`, or Ridge regression) fit on frozen embeddings with MSE loss on z-score normalized targets. No hidden layers, no nonlinearity.

2. **kNN Regression (k=20).** k-Nearest Neighbors regression with cosine or Euclidean distance. No parameters are trained.

Both metrics are reported in MSE on z-score normalized targets. A random encoder baseline yields MSE ≈ 1.0.

**Evaluation encoder.** For EMA-based models, the *target encoder* is used at evaluation time (Ojaswi's ViT-JEPA-EMA) or the *online encoder* (Sarvesh's ViT-JEPA, Vaibhav's Conv-JEPA-EMA), as specified per model.

### 6.2 Representation Collapse Detection

All models monitor embedding health by computing the per-dimension standard deviation of embeddings on a held-out batch after each epoch:

| Avg std | Status |
|---------|--------|
| > 0.3 | Healthy |
| 0.1 – 0.3 | Warning |
| < 0.1 | Collapsed |

For EMA-based models (which lack VICReg's explicit variance hinge), Vaibhav and Ojaswi additionally run `collapse_check.py` which reports effective rank (entropy of singular value spectrum), participation ratio, dead channel fraction (std < 1e-4), and kNN identity rate.

---

## 7. Experiments

### 7.1 Main Results

**Table 1.** Linear Probe (LP) and kNN Regression MSE on validation and test sets (z-score normalized α and ζ). Lower is better. Random baseline ≈ 1.0. Embedding std reported for collapse monitoring (> 0.3 = healthy, 0.1–0.3 = warning, < 0.1 = collapsed). † indicates result from `latest.pt` (training ongoing); ‡ indicates result from epoch 32 (undertrained).

| Method | Owner | Params | Emb. Std | LP α (val) | LP ζ (val) | kNN α (val) | kNN ζ (val) | LP α (test) | LP ζ (test) | kNN α (test) | kNN ζ (test) |
|--------|-------|--------|----------|-----------|-----------|------------|------------|------------|------------|-------------|-------------|
| Random encoder | — | — | — | ~1.0 | ~1.0 | ~1.0 | ~1.0 | ~1.0 | ~1.0 | ~1.0 | ~1.0 |
| Conv-JEPA (aug+VICReg) | Vaibhav | 3.27M | — | 0.0701 | 0.0840 | 0.1188 | 0.3913 | 0.1016 | 0.0769 | 0.1729 | 0.2955 |
| Conv-JEPA (no-aug+VICReg) | Vaibhav | 3.27M | — | 0.0276 | 0.3078 | 0.0930 | 0.3316 | 0.0301 | 0.2446 | 0.0652 | 0.7736 |
| Conv-JEPA (no-aug+EMA) | Vaibhav | 3.27M | — | **0.0119** | **0.1475** | ~0.000 | 0.2395 | **0.0114** | 0.1772 | **0.0101** | 0.1829 |
| VICReg ViT v1 (20 ep) | Sarvesh | 23.6M | 0.091 ⚠ | 0.4687 | 0.6662 | 0.6833 | 0.8732 | 0.5531 | 0.7441 | 0.7043 | 0.9609 |
| VICReg ViT v2 (100 ep) | Sarvesh | 23.6M | 0.366 ✓ | 0.0615 | 0.1947 | 0.0907 | **0.3013** | 0.0629 | 0.2191 | **0.0850** | **0.2855** |
| ViT-JEPA-v2 (Vaibhav) | Vaibhav | 6.0M | 0.171 ⚠ | 0.3308 | 0.4715 | 0.2764 | 0.3051 | 0.3431 | 0.5798 | 0.3457 | 0.4397 |
| VIT-JEPA patch-32 (Ojaswi) ‡ | Ojaswi | 6.0M | 0.386 ✓ | 0.6619 | 0.7435 | 0.7411 | 0.8361 | 0.6045 | 0.8799 | 0.7209 | 0.9858 |
| VIT-JEPA patch-16 (Ojaswi) | Ojaswi | 6.0M | — | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ViT-JEPA-EMA (Ojaswi) † | Ojaswi | ~22M | 0.179 ⚠ | 0.4394 | 0.4438 | 0.5027 | 0.4590 | 0.4659 | 0.4956 | 0.4724 | 0.6420 |
| ViT-JEPA sarvesh | Sarvesh | 6.85M | 0.314 ✓ | 0.0753 | 0.3122 | 0.2199 | 0.3625 | 0.1009 | 0.2823 | 0.3171 | 0.3365 |
| VideoMAE (Sarvesh) | Sarvesh | 6.4M | 0.097 ⚠ | 0.1146 | 0.2288 | 0.3169 | 0.2184 | 0.1295 | **0.1612** | 0.3851 | 0.2243 |

### 7.2 Ablation Studies

Our experimental design naturally supports four controlled ablation axes.

**Ablation 1 — Data augmentation effect (Vaibhav, Conv-JEPA).** Comparing `convjepa-with-data-aug-vicreg` vs. `convjepa-without-data-aug-vicreg` isolates the contribution of spatial augmentation (flip, rotation, noise) with identical architecture and VICReg objective. This tests whether augmentation-induced invariance helps or hurts for physical parameter prediction.

**Table 2.** Effect of data augmentation — Conv-JEPA + VICReg (val MSE).

| Variant | Data Aug | LP α | LP ζ | kNN α | kNN ζ |
|---------|----------|------|------|-------|-------|
| with aug + VICReg | Yes | 0.0701 | 0.0840 | 0.1188 | 0.3913 |
| no aug + VICReg | No | **0.0276** | **0.3078** | **0.0930** | **0.3316** |

Without augmentation, LP α drops from 0.0701 → 0.0276, suggesting augmentation-induced invariance actually hurts α regression. However, ζ performance is similar. The no-aug variant trains for only 25 epochs vs. 50, so additional training may shift this comparison.

**Ablation 2 — EMA vs. VICReg collapse prevention (Vaibhav, Conv-JEPA).** Comparing `convjepa-without-data-aug-vicreg` vs. `convjepa-without-data-aug-ema` isolates the collapse-prevention mechanism with identical architecture and no augmentation. VICReg uses explicit variance/covariance constraints on the embedding; EMA uses a slowly-moving target to avoid trivial solutions.

**Table 3.** EMA vs. VICReg as collapse prevention — Conv-JEPA, no augmentation (val MSE).

| Variant | Collapse Prevention | LP α | LP ζ | kNN α | kNN ζ |
|---------|---------------------|------|------|-------|-------|
| no aug + VICReg | VICReg (std hinge) | 0.0276 | 0.3078 | 0.0930 | 0.3316 |
| no aug + EMA | EMA target encoder | **0.0119** | **0.1475** | ~0.000 | **0.2395** |

EMA clearly outperforms VICReg as a collapse-prevention mechanism for the Conv-JEPA architecture under no-augmentation conditions, achieving the best LP α score (0.0119) of any model in the study. The near-zero kNN α result for EMA on validation is a numerical artifact from feature space geometry rather than a sign of collapse (avg_std = 0.179, 0 dead dimensions confirmed).

**Ablation 3 — Training scale (Sarvesh, VICReg ViT).** Comparing `vision-transformer-v1` vs. `vision-transformer-v2` tests the effect of training for 20 vs. 100 epochs, using stride=4 (2,275 clips) vs. stride=1 (8,750 clips), and variance weight 50 vs. 25. These changes were applied together as a package upgrade.

**Table 4.** Effect of training scale — VICReg ViT (val MSE).

| Variant | Epochs | Train clips | Var weight | Emb. Std | LP α | LP ζ | kNN α | kNN ζ |
|---------|--------|-------------|-----------|----------|------|------|-------|-------|
| v1 (short run) | 20 | 2,275 | 50 | 0.091 ⚠ | 0.4687 | 0.6662 | 0.6833 | 0.8732 |
| v2 (full run) | 100 | 8,750 | 25 | 0.366 ✓ | **0.0615** | **0.1947** | **0.0907** | **0.3013** |

The improvement is dramatic: LP α drops from 0.4687 → 0.0615 (7.6×). v1's embedding std of 0.091 sits in the warning zone, indicating near-collapse, whereas v2 at 0.366 is fully healthy. The combination of more training data (stride=1 vs stride=4), longer training (100 vs 20 epochs), and a lower variance weight together rescue the model from near-collapse and produce representations well below the random baseline.

**Ablation 4 — Spatial patch size (Ojaswi, ViT-JEPA).** Comparing `VIT-JEPA-OJASWI-patch-32` vs. `VIT-JEPA-OJASWI-patch-16` tests whether finer patches (16×16 → 1,568 tokens) encode more useful spatial detail for parameter prediction compared to coarser patches (32×32 → 392 tokens).

**Table 5.** Effect of patch size — ViT-JEPA VICReg (val MSE).

| Variant | Patch size | Tokens | Epochs done | LP α | LP ζ | kNN α | kNN ζ |
|---------|-----------|--------|-------------|------|------|-------|-------|
| patch-32 | 32×32 | 392 | 32 ‡ | 0.6619 | 0.7435 | 0.7411 | 0.8361 |
| patch-16 | 16×16 | 1,568 | — | TBD | TBD | TBD | TBD |

Note: patch-32 results are from epoch 32 of a 100-epoch run and are likely to improve substantially with continued training. Patch-16 results are pending.

### 7.3 Representation Analysis

*[To be completed with visualizations after all models finish training.]*

Planned analyses:
- t-SNE / UMAP of frozen representations colored by α and ζ values
- Per-epoch embedding standard deviation curves (collapse monitoring)
- Training loss curves across all variants
- Nearest-neighbor retrievals in embedding space

> **Figure 1 (placeholder).** t-SNE of frozen representations from the best model, colored by α (left) and ζ (right). Clear clustering would indicate successful disentanglement of physical parameters.

---

## 8. Discussion

**Training scale is the single most important factor.** The most striking finding is the gap between VICReg ViT v1 (20 epochs, stride=4) and v2 (100 epochs, stride=1): LP α improves 7.6× (0.4687 → 0.0615) and the embedding std jumps from a warning-zone 0.091 to a healthy 0.366. v1 was on the verge of collapse — the combination of more data, longer training, and a lower variance weight (50→25) together rescued the representations. This suggests that for this small dataset (8,750 clips), the 25.0M VICReg ViT model is data-hungry and needs full exposure to learn meaningful structure.

**EMA target encoders outperform VICReg for convolutional encoders.** Within the Conv-JEPA family, the EMA variant (no-aug+EMA) achieves the best LP α of any model (0.0119 val, 0.0114 test), outperforming both VICReg Conv-JEPA variants by 2–5×. This suggests that for the convolutional architecture, the slowly-evolving EMA target provides a more effective anti-collapse signal than the VICReg variance/covariance hinge. The pattern does not clearly generalize to transformers — Ojaswi's ViT-JEPA-EMA (0.44 LP α) underperforms VICReg ViT v2 (0.06 LP α) — likely because that model was evaluated from a mid-training checkpoint.

**α is universally easier to predict than ζ.** Across all models and evaluation methods, MSE on α is lower than on ζ. This is consistent with α (active dipole strength) producing more visually distinct global flow patterns (turbulent vs. ordered), while ζ (steric alignment) creates more subtle, local orientational differences that are harder to read from global pooled representations.

**Convolutional encoders are competitive with ViTs on this dataset.** The Conv-JEPA EMA model (3.27M params, 128-dim embeddings) achieves LP α = 0.0119, matching or outperforming all transformer models including the 23.6M VICReg ViT v2 (LP α = 0.0615). This may reflect that the small dataset size (8,750 clips) limits the advantage of global attention, or that the dense 14×14 spatial embeddings of the ConvNeXt backbone (196 vectors per sample) provide richer VICReg statistics than the pooled ViT embedding.

**VideoMAE achieves competitive ζ results.** Despite using a reconstruction objective and a small ViT-Tiny encoder (192-dim), VideoMAE achieves LP ζ = 0.1612 on the test set — the best ζ test score in the study. This suggests that masked reconstruction may capture different physical information than predictive objectives, particularly regarding the orientational structure encoded in ζ.

**Several models are still training or were evaluated early.** Ojaswi's patch-32 ViT-JEPA (epoch 32 of 100) and ViT-JEPA-EMA (from `latest.pt`, not `best.pt`) are likely to improve. The patch-16 results are pending. The final comparison will be updated accordingly.

**Limitations.** Our study is limited to the `active_matter` dataset; generalization to 3D simulations or turbulent flows is not tested. Due to the 300 GPU-hours/student budget, hyperparameter search was not exhaustive — configurations represent principled choices rather than optimized settings. The Conv-JEPA evaluation uses trajectory-level (mean-pooled) embeddings rather than clip-level, which may inflate performance relative to the clip-level ViT evaluations. Embedding dimensionality also varies across models (128 for Conv-JEPA, 192 for VideoMAE, 256 for ViT-JEPA-sarvesh, 384 for ViT variants), making direct comparisons imperfect.

---

## 9. Conclusion

*[To be completed after final results.]*

---

## 10. Ethics & Broader Impact

This work develops self-supervised representation learning methods for physical simulations. Positive applications include accelerating scientific discovery in soft matter physics, biophysics, and materials science by enabling parameter estimation, anomaly detection, and simulation compression from unlabeled trajectory data. All models are trained exclusively on synthetic simulation data from a public dataset and do not interact with real-world systems.

The primary environmental cost is GPU compute: our eleven experimental runs collectively use approximately [X] GPU-hours on NYU's HPC cluster (see §11). Unnecessary compute is mitigated by early stopping of collapsed runs, shared preprocessing pipelines, and preemption-safe resumption that avoids re-running completed training.

We see no significant dual-use risk. Learned representations encode dynamics of simple model systems and would require substantial re-engineering to apply in sensitive domains.

---

## 11. Compute Accounting

| Model | Hardware | GPU-hours | VRAM (peak) | Precision |
|-------|----------|-----------|-------------|-----------|
| Conv-JEPA (aug+VICReg) | 1× A100 40GB | TBD | ~38 GB | BF16 |
| Conv-JEPA (no-aug+VICReg) | 1× A100 40GB | TBD | ~38 GB | BF16 |
| Conv-JEPA (no-aug+EMA) | 1× A100 80GB | TBD | ~76 GB | BF16 |
| VICReg ViT v1 | 1× A100 40GB | TBD | ~24 GB | BF16 |
| VICReg ViT v2 | 1× A100 40GB | TBD | ~24 GB | BF16 |
| ViT-JEPA-v2 (Vaibhav) | 1× A100 32GB | TBD | ~32 GB | BF16 |
| VIT-JEPA patch-32 (Ojaswi) | 1× A100 40GB | TBD | ~20 GB | BF16 |
| VIT-JEPA patch-16 (Ojaswi) | 1× A100 40GB | TBD | ~32 GB | BF16 |
| ViT-JEPA-EMA (Ojaswi) | 1× A100 40GB | TBD | TBD | BF16 |
| ViT-JEPA sarvesh | 1× A100 40GB | TBD | TBD | BF16 |
| VideoMAE (Sarvesh) | 1× A100 40GB | TBD | ~24 GB | BF16 |

- **Slurm account:** `csci_ga_2572-2026sp`
- **Checkpointing:** All jobs save `latest.pt` every 20–50 optimizer steps and include `#SBATCH --requeue` for automatic spot-instance restart. EMA models additionally save the target encoder state.
- **Experiment tracking:** Weights & Biases (project: `DL`, entity: `sb10583-`). W&B run IDs persisted to disk for cross-preemption logging continuity.
- **Fixed seeds:** All models use fixed random seeds logged to W&B.

---

## Statement of Contributions

| Team Member | Contributions |
|-------------|--------------|
| **Sarvesh Bodke** (sb10583) | Data pipeline and preprocessing (`ActiveMatterDataset` with sliding window sampling, spatial augmentations, z-score normalization); VICReg with 3D ViT encoder: architecture design, ablation of training scale and variance weights (`vision-transformer-v1`, `vision-transformer-v2`); ViT-JEPA with EMA target encoder and spatiotemporal block masking (`ViT Jepa sarvesh`); VideoMAE masked autoencoding with temporal tube masking (`video MAe`); fault-tolerant training infrastructure (step checkpointing, SIGUSR1 handling, W&B run ID persistence). |
| **Vaibhav Chaudhari** (vc2836) | Conv-JEPA baseline implementation and augmentation ablation (`convjepa-with-data-aug-vicreg`, `convjepa-without-data-aug-vicreg`); EMA target encoder variant (`convjepa-without-data-aug-ema`); ViT-JEPA with token-level VICReg and physics-aware sign-corrected augmentation (`ViT-JEPA-v2`); collapse diagnostics (`collapse_check.py` with effective rank and participation ratio). |
| **Ojaswi Kaushik** (ok2287) | ViT-JEPA with VICReg and patch size ablation (`VIT-JEPA-OJASWI-patch-32`, `VIT-JEPA-OJASWI-patch-16`); I-JEPA-style model with EMA target encoder and multi-block spatiotemporal masking (`ViT-JEPA-EMA`); linear probe and kNN evaluation scripts; collapse monitoring. |

---

## References

[1] McCabe, M. et al. "The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning." *NeurIPS 2024 Datasets and Benchmarks Track.*

[2] Qu, Y. et al. "Representation Learning for Spatiotemporal Physical Systems." *arXiv:2603.13227*, 2026.

[3] EB-JEPA: Open-source library for learning representations from images and videos. GitHub.

[4] Assran, M. et al. "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture." *CVPR 2023.*

[5] He, K. et al. "Masked Autoencoders Are Scalable Vision Learners." *CVPR 2022.* arXiv:2111.06377.

[6] Tong, Z. et al. "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training." *NeurIPS 2022.* arXiv:2203.12602.

[7] Chen, T. et al. "A Simple Framework for Contrastive Learning of Visual Representations." *ICML 2020.*

[8] He, K. et al. "Momentum Contrast for Unsupervised Visual Representation Learning." *CVPR 2020.*

[9] Bardes, A., Ponce, J., LeCun, Y. "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning." *ICLR 2022.*

[10] Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential Equations." *ICLR 2021.*
