# Self-Supervised Representation Learning for Active Matter Physical Simulations

**CSCI-GA 2572 Deep Learning — Spring 2026**  
**NYU Courant Institute of Mathematical Sciences**

**Team:** Sarvesh Bodke (sb10583) · Vaibhav Chaudhari (vc2836) · Ojaswi Kaushik (ok2287)

---

## Abstract

> *[Write after final results are in. Suggested structure: "We present a systematic study of self-supervised representation learning methods applied to the active_matter physical simulation dataset. We explore five families of approaches — convolutional JEPA, VICReg with a 3D Vision Transformer, ViT-JEPA with temporal predictive coding, ViT-JEPA with EMA target encoders and block masking, and VideoMAE masked autoencoding — and evaluate each by how well frozen representations predict the underlying physical parameters α and ζ via linear probing and kNN regression. Our best model achieves a linear probing MSE of [X] / [X] and kNN MSE of [X] / [X] on normalized α and ζ respectively. We find that [key insight — e.g., global attention outperforms local convolutions / reconstruction objectives encode complementary structure to predictive objectives]."]*

---

## 1. Introduction

Physical simulations of active matter systems describe the collective dynamics of rod-like active particles immersed in a viscous fluid — a canonical model for systems ranging from bacterial colonies to cytoskeletal networks. The temporal evolution of such systems is governed by two underlying physical parameters: **α** (active dipole strength), controlling energy injection by active units, and **ζ** (steric alignment), controlling dissipation. Learning compact, physically meaningful representations of these dynamics without access to these labels is a fundamental challenge at the intersection of machine learning and physical sciences.

We study **self-supervised representation learning** on the `active_matter` dataset from The Well [1], a large-scale benchmark for physics-informed machine learning. The dataset provides spatiotemporal trajectories of active matter simulations as 11-channel physical fields (concentration, velocity, orientation, strain-rate) evolving over time. Our goal is to pre-train an encoder using self-supervised objectives and evaluate the quality of the learned representations by predicting α and ζ using only a frozen encoder followed by a linear layer or kNN regressor.

A key observation that motivates our approach is the structural analogy between this dataset and video data: just as a video is a sequence of 2D RGB frames (T × H × W × 3), an active matter simulation is a sequence of 2D multi-channel physical fields (T × H × W × 11). This analogy allows us to directly adapt and compare state-of-the-art video self-supervised learning architectures — VideoMAE [6], ViT-JEPA [4] — to the physical simulation domain.

Our contributions are:
- A systematic comparison of five families of self-supervised objectives (contrastive-free VICReg, JEPA latent prediction, and masked autoencoding) on physical simulation data.
- An evaluation of convolutional versus transformer encoders for capturing the long-range spatiotemporal patterns that determine α and ζ.
- Ablation studies over patch size, training duration, data augmentation, and collapse-prevention strategy (VICReg variance term vs. EMA target encoder).
- A complete reproducible pipeline — training, checkpointing, and linear/kNN evaluation — running on NYU HPC with Weights & Biases logging.

---

## 2. Related Work

**Self-Supervised Representation Learning.**
Contrastive methods such as SimCLR [7] and MoCo [8] learn representations by pulling together embeddings of augmented views of the same sample while pushing apart different samples. VICReg [9] removes the need for negative pairs by instead regularizing embedding variance and covariance across dimensions, preventing collapse without contrastive sampling. Masked Autoencoders (MAE) [5] take a reconstruction-based approach, training a ViT encoder to reconstruct randomly masked image patches through a lightweight decoder. VideoMAE [6] extends this to video by applying temporal tube masking at high masking ratios (90%), forcing the model to understand genuine spatiotemporal dynamics rather than interpolating between nearby frames.

**JEPA-style Predictive Architectures.**
I-JEPA [4] proposes learning representations by predicting the latent embeddings of masked target regions from visible context patches — operating in representation space rather than pixel space. This avoids the need to reconstruct low-level texture and instead focuses on high-level semantic structure. V-JEPA extends this to video by predicting spatiotemporal target blocks. JEPA-style methods typically use an EMA (exponential moving average) target encoder to provide a stable prediction target, preventing collapse without requiring contrastive negatives.

**Representation Learning for Physical Systems.**
Qu et al. [2] introduce a convolutional JEPA baseline for spatiotemporal physical simulations, using a ConvNeXt encoder to predict latent representations of future frames from context frames under a VICReg objective. This work, which directly informs our baseline, demonstrates that self-supervised predictive coding can capture physically meaningful structure in simulated systems. Neural operators and learned simulators [10] take a related but distinct approach, directly learning the evolution operator of PDEs rather than representations for downstream tasks.

**Video Representation Learning.**
The structural analogy between video (T × H × W × 3) and physical simulation (T × H × W × 11) data allows direct transfer of video architectures. Key insights from the video domain — tube masking to prevent trivial temporal interpolation [6], attention over full spatiotemporal token sequences [4], and EMA target encoders for stable training [4] — all translate naturally to the physical simulation setting we study.

---

## 3. Dataset

We use the **`active_matter`** subset from The Well [1], a large-scale dataset of physical simulations hosted on HuggingFace (`polymathic-ai/active_matter`, ~52 GB).

**Physical System.** Each simulation models a 2D active matter system — rod-like active particles immersed in a viscous fluid — evolving under the equations of active nematic liquid crystal theory. The dynamics are governed by two physical constants: **α** (active dipole strength, 5 discrete values) and **ζ** (steric alignment, 9 discrete values), yielding 45 unique parameter combinations.

**Data Structure.** Each HDF5 file contains 3 independent simulation trajectories of 81 time steps each at spatial resolution 256×256. Each frame has 11 physical channels corresponding to 5 tensor fields:

| Field | Key | Channels | Description |
|-------|-----|----------|-------------|
| Concentration | `t0_fields/concentration` | 1 | Scalar density field |
| Velocity | `t1_fields/velocity` | 2 | 2D velocity vector field |
| Orientation tensor | `t2_fields/D` | 4 | Nematic order tensor |
| Strain-rate tensor | `t2_fields/E` | 4 | Symmetric strain-rate tensor |

**Dataset Splits.**

| Split | Files | Samples |
|-------|-------|---------|
| Train | 45 | 8,750 |
| Validation | 16 | 1,200 |
| Test | — | 1,300 |

**Labels.** α and ζ are withheld during pre-training and used **only** for downstream evaluation (linear probing and kNN regression), normalized via z-score. The task is framed as regression (MSE loss), not classification.

**Preprocessing.** All models apply a common preprocessing pipeline:
1. Extract a sliding window of 16 (or 32) consecutive frames per sample.
2. Apply per-sample, per-channel z-score normalization across the temporal and spatial dimensions.
3. Random 224×224 crop from 256×256 (train) / center crop (val, test).
4. Data augmentation (train only): random horizontal/vertical flip and 90° rotation, which are valid isotropic symmetries of the simulation domain.
5. Permute to `(C, T, H, W) = (11, 16, 224, 224)`.

Some models additionally apply Gaussian noise augmentation (σ = 1.0) at the input level during training.

---

## 4. Methods

We explore five families of self-supervised architectures. All models are trained **from scratch** with no pretrained weights, stay under the 100M parameter limit, and use only the training split for weight updates.

### 4.1 Baseline: Convolutional JEPA (Conv-JEPA)

```
Context (B,11,16,224,224) ──► ConvNeXt (3D→2D) ──► z_ctx (B,128,H',W')
                                                             │
                                                      Conv Predictor
                                                      (C → 2C → C)
                                                             │
                                                      z_pred (B,128,H',W')
                                                             ║  VICReg
Target  (B,11,16,224,224) ──► ConvNeXt (3D→2D) ──► z_tgt  (B,128,H',W')
```

Following Qu et al. [2], our baseline uses a **5-stage ConvNeXt encoder** with channel dimensions [16, 32, 64, 128, 128] and residual blocks [3, 3, 3, 9, 3]. The encoder uses 3D convolutions with per-stage 2× downsampling until the temporal dimension collapses to 1, after which it continues with 2D residual blocks. A shallow convolutional predictor head (C → 2C → C) maps the context encoding to a prediction of the target encoding.

**Training objective.** Given a 16-frame context clip and a subsequent 16-frame target clip, the encoder maps both to dense spatial embeddings. VICReg loss [9] is applied between the predicted and actual target embeddings with weights sim=2, std=40, cov=2. The high variance weight (40) is critical to prevent collapse — the encoder is trained without an EMA target or negative pairs.

**Ablations (Vaibhav).** Three variants of Conv-JEPA were trained to ablate the effect of data augmentation and collapse-prevention strategy:

| Variant | Data Aug | Collapse Prevention |
|---------|----------|---------------------|
| `jepa-baseline` | Yes (flip, rotation, noise) | VICReg |
| `convjepa-without-data-aug-vicreg` | No | VICReg |
| `convjepa-without-data-aug-ema` | No | EMA target encoder |

| Component | Parameters |
|-----------|-----------|
| ConvNeXt Encoder | ~2.47M |
| Conv Predictor | ~0.80M |
| **Total** | **~3.3M** |

### 4.2 VICReg with 3D Vision Transformer Encoder

```
View 1 (B,11,16,224,224) ──► Conv3D(2×32×32) ──► 392 tokens ──► ViT(6L,384) ──► Pool ──► (B,384)
                                                                                                │
                                                                                         Projector MLP
                                                                                      (384→2048→2048→2048)
                                                                                                │
                                                                                            z1 (B,2048)
                                                                                                ║ VICReg
View 2 (B,11,16,224,224) ──► Conv3D(2×32×32) ──► 392 tokens ──► ViT(6L,384) ──► Pool ──► z2 (B,2048)

  [Projector discarded at eval — only ViT encoder + Pool used for representations]
```

Motivated by the need for global receptive fields to capture the large-scale spatiotemporal patterns that determine α and ζ, we replace the ConvNeXt encoder with a **3D Vision Transformer (ViT)**. The 3D patch embedding uses a Conv3D with kernel `(2, 32, 32)` and stride `(2, 32, 32)`, converting the input `(B, 11, 16, 224, 224)` into 392 spatiotemporal tokens `(8T × 7H × 7W)`. Fixed 3D sinusoidal positional embeddings encode each token's position along the temporal, height, and width axes.

**Architecture.** The ViT encoder consists of 6 transformer blocks (embed_dim=384, 6 heads, MLP ratio 4×, pre-norm). After encoding, tokens are mean-pooled to a single 384-dimensional embedding. A 3-layer MLP projector (384 → 2048 → 2048 → 2048) is appended during training and discarded at evaluation.

**Training objective.** VICReg loss is applied between projector outputs of two temporally offset views (context and target frames). Two variants were trained to ablate the variance weight:

| Variant | Epochs | Stride | Var Weight | Batch (eff.) |
|---------|--------|--------|-----------|--------------|
| `vision-transformer-v1` | 20 | 4 (2,275 clips) | 50 | 32 |
| `vision-transformer-v2` | 100 | 1 (8,750 clips) | 25 | 64 |

v2 additionally introduced step-level checkpointing every 20 steps and SIGUSR1 preemption handling for fault tolerance on cloud spot instances.

| Component | Parameters |
|-----------|-----------|
| ViT Encoder (6L, dim=384) | ~11M |
| MLP Projector (384→2048→2048→2048) | ~12.6M |
| **Total** | **~23.6M** |

### 4.3 ViT-JEPA with Temporal Predictive Coding

```
Context frames (B,11,16,224,224)            Target frames (B,11,16,224,224)
        │                                           │
  PatchEmbed3D                               PatchEmbed3D
 (Conv3D 2×32×32)                           (Conv3D 2×32×32)
        │                                           │
  392 tokens                                  392 tokens
        │                                           │
  ViT Encoder (6L, 384)                      ViT Encoder (6L, 384) [shared weights]
  (B, 392, 384)                               (B, 392, 384)
        │                                           │
  Transformer Predictor                       Global Avg Pool
  (2L, 384→192→384)                                │
  + Global Avg Pool                           z_tgt (B, 384)
        │                                           ║
  z_pred (B, 384) ══════════ VICReg ════════════════╝
```

Rather than applying VICReg directly between two pooled embeddings, this family of methods uses a **shallow transformer predictor** operating on the full token sequence. The encoder maps context frames to 392 tokens; the predictor takes the full token sequence, projects it to a lower-dimensional bottleneck, applies 2 transformer blocks, and projects back — producing a predicted target embedding. VICReg is applied between the pooled predictor output and the pooled target encoder output.

**Key advantage.** Operating on 392 tokens (rather than a pooled vector) allows the predictor to attend over specific spatial regions and time steps before summarizing, preserving spatiotemporal structure that would be lost after pooling.

**Vaibhav's ViT-JEPA-v2.** Uses the same 3D patch embedding and ViT encoder as above (depth=6, dim=384, patch=32), with a 2-layer transformer predictor (dim=192 bottleneck). VICReg weights: sim=2, std=40, cov=2.

**Ojaswi's ViT-JEPA variants (patch-32 and patch-16).** Two variants ablate spatial patch size:
- `VIT-JEPA-OJASWI-patch-32`: 32×32 patches → 392 tokens
- `VIT-JEPA-OJASWI-patch-16`: 16×16 patches → 1,568 tokens (4× more tokens, finer features, matches resolution used in Qu et al. [2])

Both use VICReg loss with a shallow 2-layer transformer predictor on the full token sequence.

| Component | Patch-32 | Patch-16 |
|-----------|---------|---------|
| ViT Encoder (6L, dim=384) | ~5.3M | ~5.3M |
| Transformer Predictor (2L, dim=192) | ~0.6M | ~0.6M |
| **Total** | **~6.0M** | **~6.0M** |

### 4.4 ViT-JEPA with EMA Target Encoder and Block Masking

```
Full clip (B,11,16,224,224)
        │
  PatchEmbed3D ──► all N tokens
        │                  │
  Block Mask               │ (no gradient)
  (~75% visible)           ▼
        │           Target Encoder  ◄── EMA update (momentum m → 0.9999)
        │           (θ_tgt = m·θ_tgt + (1-m)·θ_online)
        │                  │
  Visible tokens    Target embeddings at
        │           masked positions
  Online Encoder          │
  (θ_online, grad)        │
        │                  │
  Predictor               │
  (narrow ViT,            │
   masked positions)       │
        │                  │
  pred (B, n_mask, d) ══ MSE ══ tgt (B, n_mask, d)   [L2-normalized]
```

This family adds two ingredients that further stabilize training: an **EMA (Exponential Moving Average) target encoder** and **spatiotemporal block masking**.

**EMA target encoder.** Instead of using the same encoder for both context and target (as in §4.3), we maintain a separate target encoder whose weights are a running average of the online encoder: θ_target ← m · θ_target + (1−m) · θ_online. The momentum m is cosine-annealed from 0.996 to 0.9999 over training. The target encoder sees the full (unmasked) clip, providing a stable, collapsed-free training signal without requiring VICReg regularization.

**Block masking.** Rather than the temporal context/target split used in §4.1–4.3, block masking randomly selects contiguous 3D blocks of patches to mask. The online encoder sees only the remaining visible patches; the predictor maps context tokens to predictions of the target encoder's output at the masked positions.

**Ojaswi's ViT-JEPA-EMA.** Uses a ViT-Small context encoder (12 layers, 6 heads, dim=384) with the corresponding EMA target encoder. The predictor is a narrow ViT bottleneck (6 layers, dim=192). To reduce memory, the temporal dimension is subsampled to 4 frames with 32×32 spatial patches. Multi-block masking (4 target blocks, 1 context block) is extended into temporal tubes.

**Sarvesh's ViT-JEPA (sarvesh).** Uses a 6-block online encoder (dim=256, 8 heads) with patch size 16 → 1,568 tokens. The EMA target encoder uses cosine-annealed momentum from 0.996 to 0.9999. Spatiotemporal block masking covers ~75% of context and ~25% target. The training objective is MSE between L2-normalized predictor output and target encoder output (no VICReg). ~6.85M parameters total.

| Hyperparameter | Ojaswi (ViT-JEPA-EMA) | Sarvesh (ViT-JEPA) |
|----------------|-----------------------|--------------------|
| Patch size | 32 | 16 |
| Tokens | 4T × 7² = 196 | 8T × 14² = 1,568 |
| Encoder depth | 12 | 6 |
| Embed dim | 384 | 256 |
| Momentum schedule | 0.996 → 1.0 | 0.996 → 0.9999 |
| Loss | MSE (L2-norm) | MSE (L2-norm) |
| Parameters | ~22M | ~6.85M |

### 4.5 VideoMAE: Masked Autoencoding

VideoMAE [6] learns representations through **reconstruction** of masked spatiotemporal patches rather than latent prediction.

```
Input (B,11,16,224,224)
        │
  Conv3D(2×16×16) ──► 1,568 tokens  (8T × 14H × 14W)
        │
  Temporal Tube Masking (90%)
  ┌──────────────────────────────────────┐
  │  Spatial grid 14×14 = 196 positions │
  │  Keep 10% visible = 19 positions    │
  │  Same spatial mask across all 8T    │  → "tubes"
  └──────────────────────────────────────┘
        │                    │
  152 visible tokens   1,416 masked tokens
        │                    │  (dropped from encoder)
  ViT-Tiny Encoder           │
  (12L, dim=192)             │
  (B, 152, 192)              │
        │                    │
        └──────────┬──────────┘
                   ▼
             MAE Decoder
       (project 192→96, fill [MASK] tokens,
        4L transformer, linear head 96→5,632)
                   │
             (B, 1568, 5632)
                   │
         MSE on masked patches only
         (per-patch normalized targets)

  [Decoder discarded at eval — only ViT-Tiny encoder used]
``` We adapt this directly to the 11-channel physical simulation setting.

**Tubelet embedding.** The input `(B, 11, 16, 224, 224)` is tokenized via a Conv3D with kernel `(2, 16, 16)` → 1,568 spatiotemporal tokens (8T × 14H × 14W). Each token represents a 2×16×16 spatiotemporal tube.

**Temporal tube masking.** 90% of tokens are masked using a tube masking strategy: a random set of spatial positions (14×14=196, of which only 10% = 19 are kept visible) is selected, and the same spatial mask is applied across all 8 time steps. This forces the model to reason about genuine spatiotemporal dynamics — not simply interpolate between nearby frames — making the 90% ratio tractable.

**Encoder (ViT-Tiny).** The encoder operates **only on the 152 visible tokens** (19 spatial × 8 temporal), consisting of 12 transformer blocks (dim=192, 3 heads). Encoder parameters: ~6.4M.

**Decoder (lightweight, training only).** The decoder reconstructs all 1,568 patches:
1. Project encoder output 192 → 96 via linear layer.
2. Fill masked positions with a learned `[MASK]` token.
3. Add full 3D sinusoidal positional embeddings to all 1,568 positions.
4. Apply 4 lightweight transformer blocks (dim=96, 3 heads).
5. Linear prediction head: 96 → 5,632 (= 2×16×16×11 per patch).

The decoder (~1.0M parameters) is discarded after training.

**Training objective.** MSE over masked patches only, with per-patch normalization of the target to remove local mean/variance bias — following the original MAE formulation [5].

$$\mathcal{L}_{\text{MAE}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \left\| \hat{x}_i - \frac{x_i - \mu_i}{\sigma_i} \right\|_2^2$$

where $\mathcal{M}$ is the set of masked patch indices, $\hat{x}_i$ is the decoder's prediction, and $\mu_i, \sigma_i$ are the per-patch mean and standard deviation of the ground-truth patch.

| Component | Parameters |
|-----------|-----------|
| ViT-Tiny Encoder (12L, dim=192) | ~6.4M |
| MAE Decoder (4L, dim=96) | ~1.0M |
| **Total (training)** | **~7.4M** |
| **Encoder only (eval)** | **~6.4M** |

---

## 5. Evaluation Protocol

### 5.1 Frozen Encoder Evaluation

After pre-training, encoder weights are **fully frozen** (no fine-tuning). Representations are extracted for all samples in the training, validation, and test splits by passing each clip through the encoder and applying global average pooling over the token sequence.

Two evaluation methods are applied to predict α and ζ **independently** as continuous regression targets (z-score normalized):

1. **Linear Probing (Ridge Regression).** A single linear layer (`nn.Linear(d → 1)`) or Ridge regression fit on the frozen embeddings. MSE loss on z-score normalized targets. No hidden layers, no activation functions.

2. **kNN Regression (k = 20).** k-Nearest Neighbors regression with cosine similarity (Ojaswi, Vaibhav, ViT-JEPA-v2) or Euclidean distance (Sarvesh VICReg, VideoMAE), k=20. No model parameters are trained.

We report MSE on the **validation set** for model selection and on the **test set** for final results. A random encoder (untrained weights) provides a baseline MSE of approximately 1.0 on z-scored targets.

### 5.2 Representation Collapse Detection

All models monitor embedding health after each epoch by computing the per-dimension standard deviation of embeddings on a held-out batch:

| std range | Status |
|-----------|--------|
| < 0.1 | Collapsed — stop training |
| 0.1 – 0.3 | Warning |
| > 0.3 | Healthy |

For VICReg models, the std loss component directly penalizes collapsed dimensions. For EMA models, collapse is prevented by the momentum-averaged target encoder without an explicit variance penalty.

### 5.3 Supervised Baseline

As a reference point, an end-to-end supervised model trained directly on (α, ζ) labels establishes an approximate upper bound on what the representations could achieve. Results TBD.

---

## 6. Experiments

### 6.1 Main Results

*Training is ongoing for several models. The table below will be updated with final test-set results.*

**Table 1.** Linear Probe and kNN Regression MSE on test set (z-score normalized α and ζ). Lower is better. Random baseline ≈ 1.0.

| Method | Encoder | Params | LP MSE (α) | LP MSE (ζ) | kNN MSE (α) | kNN MSE (ζ) |
|--------|---------|--------|-----------|-----------|------------|------------|
| Random encoder | — | — | ~1.0 | ~1.0 | ~1.0 | ~1.0 |
| Conv-JEPA (jepa-baseline) | ConvNeXt | 3.3M | TBD | TBD | TBD | TBD |
| Conv-JEPA (no aug, VICReg) | ConvNeXt | 3.3M | TBD | TBD | TBD | TBD |
| Conv-JEPA (no aug, EMA) | ConvNeXt | 3.3M | TBD | TBD | TBD | TBD |
| VICReg ViT v1 (20 ep) | ViT (dim=384, p=32) | 23.6M | TBD | TBD | TBD | TBD |
| VICReg ViT v2 (100 ep) | ViT (dim=384, p=32) | 23.6M | TBD | TBD | TBD | TBD |
| ViT-JEPA-v2 (Vaibhav) | ViT (dim=384, p=32) | 6.0M | TBD | TBD | TBD | TBD |
| VIT-JEPA patch-32 (Ojaswi) | ViT (dim=384, p=32) | 6.0M | TBD | TBD | TBD | TBD |
| VIT-JEPA patch-16 (Ojaswi) | ViT (dim=384, p=16) | 6.0M | TBD | TBD | TBD | TBD |
| ViT-JEPA-EMA (Ojaswi) | ViT-Small (dim=384) | ~22M | TBD | TBD | TBD | TBD |
| ViT-JEPA sarvesh (EMA) | ViT (dim=256, p=16) | 6.85M | TBD | TBD | TBD | TBD |
| VideoMAE (Sarvesh) | ViT-Tiny (dim=192) | 6.4M | TBD | TBD | TBD | TBD |

### 6.2 Ablation Studies

Our experimental design supports several natural ablation comparisons across models.

**Effect of training scale (Sarvesh VICReg v1 vs v2).** We increase training from 20 to 100 epochs and change the dataset sampling stride from 4 (2,275 clips) to 1 (8,750 clips), giving the model ~4× more unique training samples per epoch. The variance weight is also reduced from 50 to 25 as more diverse samples reduce collapse risk.

**Table 2.** Effect of training scale — VICReg ViT variants (val set).

| Variant | Epochs | Stride | Var Weight | LP MSE (α) | LP MSE (ζ) | kNN MSE (α) | kNN MSE (ζ) |
|---------|--------|--------|-----------|-----------|-----------|------------|------------|
| v1 (short) | 20 | 4 | 50 | TBD | TBD | TBD | TBD |
| v2 (full) | 100 | 1 | 25 | TBD | TBD | TBD | TBD |

**Effect of patch size (Ojaswi ViT-JEPA patch-32 vs patch-16).** Smaller patches (16×16) produce 4× more tokens (1,568 vs 392) and capture finer-grained spatial structure, at the cost of longer attention sequences.

**Table 3.** Effect of patch size — ViT-JEPA (Ojaswi) variants (val set).

| Variant | Patch Size | Tokens | LP MSE (α) | LP MSE (ζ) | kNN MSE (α) | kNN MSE (ζ) |
|---------|-----------|--------|-----------|-----------|------------|------------|
| patch-32 | 32×32 | 392 | TBD | TBD | TBD | TBD |
| patch-16 | 16×16 | 1,568 | TBD | TBD | TBD | TBD |

**Effect of data augmentation (Vaibhav Conv-JEPA).** Comparing Conv-JEPA with and without spatial augmentations (flip, rotation, Gaussian noise) tests whether invariance to these isotropic symmetries helps or hurts representation quality for parameter regression.

**Table 4.** Effect of data augmentation — Conv-JEPA variants (val set).

| Variant | Data Aug | LP MSE (α) | LP MSE (ζ) | kNN MSE (α) | kNN MSE (ζ) |
|---------|----------|-----------|-----------|------------|------------|
| jepa-baseline | Yes | TBD | TBD | TBD | TBD |
| without-aug-vicreg | No | TBD | TBD | TBD | TBD |

**Effect of collapse-prevention (EMA vs VICReg, Vaibhav Conv-JEPA).** Both Conv-JEPA without-aug-vicreg and without-aug-ema have identical architecture and data; they differ only in whether collapse is prevented by VICReg's variance term or an EMA target encoder.

**Table 5.** EMA vs VICReg as collapse-prevention — Conv-JEPA (no aug) variants (val set).

| Variant | Collapse Prevention | LP MSE (α) | LP MSE (ζ) | kNN MSE (α) | kNN MSE (ζ) |
|---------|---------------------|-----------|-----------|------------|------------|
| without-aug-vicreg | VICReg (std term) | TBD | TBD | TBD | TBD |
| without-aug-ema | EMA target encoder | TBD | TBD | TBD | TBD |

**Effect of objective: predictive coding vs reconstruction.** Comparing ViT-JEPA variants (latent prediction) against VideoMAE (pixel-space reconstruction) on the same ViT backbone tests whether physical parameter information is better captured by predictive or reconstructive pre-training.

### 6.3 Representation Analysis

*[To be completed with final results.]*

- t-SNE / UMAP visualizations of frozen embeddings colored by α and ζ values.
- Embedding standard deviation curves over training epochs (collapse monitoring).
- Training loss curves for all models.

> **Figure 1:** *(placeholder)* t-SNE of frozen representations from the best-performing model, colored by α (left) and ζ (right). Clear clustering by parameter value would indicate successful disentanglement of physical parameters.

---

## 7. Discussion

*[To be completed after final results.]*

Key questions to address:
- Do transformer-based encoders outperform convolutional encoders for capturing the global, long-range patterns that determine α and ζ?
- Does reconstruction (VideoMAE) or latent prediction (ViT-JEPA) yield better representations for physical parameter regression?
- Does EMA or VICReg more reliably prevent collapse across architectures?
- Does finer patch granularity (patch-16 vs patch-32) help when the relevant physical patterns span multiple scales?

**Limitations.** Our study is limited to the `active_matter` dataset and thus a specific class of 2D active nematic simulations; generalization to other physical systems (3D, turbulent flow, etc.) is not tested. Due to GPU hour constraints (300 hours/student on the NYU HPC), full hyperparameter search was not feasible — reported configurations represent principled choices rather than exhaustive optimization. Several models are still training at time of writing and results will be updated.

---

## 8. Conclusion

*[To be completed after final results.]*

---

## 9. Ethics & Broader Impact

This work develops general-purpose representation learning methods for physical simulations. Positive impacts include accelerating scientific discovery in soft matter physics, biophysics, and materials science by enabling downstream tasks (parameter estimation, anomaly detection, simulation compression) from unlabeled simulation data. The methods are trained exclusively on synthetic simulation data and do not interact with real-world systems.

The primary environmental cost is GPU compute: our experiments collectively use approximately [X] GPU-hours on NYU's HPC cluster (see §10). We mitigate unnecessary compute by sharing preprocessing pipelines, enabling checkpoint resumption from preemption, and stopping clearly collapsed runs early.

We see no significant dual-use risk: the representations encode physical dynamics of simple model systems and do not generalize to sensitive domains without substantial re-engineering.

---

## 10. Compute Accounting

| Approach | Hardware | Approx. GPU-hours | VRAM Peak | Precision |
|----------|----------|-------------------|-----------|-----------|
| Conv-JEPA variants (×3) | 1× A100 40GB | TBD | ~38 GB | BF16 |
| VICReg ViT v1 | 1× A100 40GB | TBD | ~24 GB | BF16 |
| VICReg ViT v2 | 1× A100 40GB | TBD | ~24 GB | BF16 |
| ViT-JEPA-v2 (Vaibhav) | 1× A100 40GB | TBD | ~20 GB | BF16 |
| VIT-JEPA patch-32/16 (Ojaswi) | 1× A100 40GB | TBD | ~20 GB | BF16 |
| ViT-JEPA-EMA (Ojaswi) | 1× A100 40GB | TBD | TBD | BF16 |
| ViT-JEPA sarvesh | 1× A100 40GB | TBD | TBD | BF16 |
| VideoMAE (Sarvesh) | 1× A100 40GB | TBD | ~24 GB | BF16 |

- **Slurm account:** `csci_ga_2572-2026sp`
- **Checkpointing:** All jobs save `latest.pt` every 20–50 optimizer steps and include `#SBATCH --requeue` for automatic restart on spot preemption. W&B run IDs are persisted to disk so logging continues seamlessly across preemptions.
- **Experiment tracking:** Weights & Biases (project: `DL` under entity `sb10583-`).
- **Seeds:** Fixed random seeds used for reproducibility; seeds logged to W&B.

---

## Statement of Contributions

| Team Member | Contributions |
|-------------|--------------|
| **Sarvesh Bodke** (sb10583) | Data pipeline and preprocessing (`ActiveMatterDataset` with sliding window, augmentations, z-score normalization); VICReg with 3D ViT encoder (vision-transformer-v1, v2); ViT-JEPA with EMA target encoder and spatiotemporal block masking (ViT Jepa sarvesh); VideoMAE masked autoencoding (video MAe); fault-tolerant training infrastructure (step checkpointing, SIGUSR1 handling, W&B run ID persistence). |
| **Vaibhav Chaudhari** (vc2836) | Conv-JEPA baseline implementation (jepa-baseline, convjepa-without-data-aug-vicreg, convjepa-without-data-aug-ema); ablation over data augmentation and EMA vs VICReg collapse prevention; ViT-JEPA with shallow transformer predictor and temporal predictive coding (ViT-JEPA-v2). |
| **Ojaswi Kaushik** (ok2287) | ViT-JEPA with VICReg objective and patch size ablation (VIT-JEPA-OJASWI-patch-32, VIT-JEPA-OJASWI-patch-16); ViT-JEPA with EMA target encoder and multi-block spatial masking (ViT-JEPA-EMA); linear probe and kNN evaluation scripts. |

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
