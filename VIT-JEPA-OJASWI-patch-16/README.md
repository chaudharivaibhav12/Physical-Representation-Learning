## **Vision Transformer — ViT-JEPA on Active Matter**

[cite_start]Self-supervised representation learning on physical simulations using a **Joint-Embedding Predictive Architecture (JEPA)** [cite: 9] combined with a **3D Vision Transformer (ViT)**. [cite_start]The model is trained to predict the latent representations of future states from current ones using a self-supervised **VICReg** loss[cite: 10].

---

### **Architecture Overview**

**Training Pipeline:**
* **Context Path**: `context` clip (first 16 frames, 11 channels) → **3D ViT Encoder** → token sequence `(B, 1568, 384)` → **Shallow Transformer Predictor** → pooled `z_pred` `(B, 384)`.
* **Target Path**: `target` clip (next 16 frames, 11 channels) → **SAME 3D ViT Encoder** → token sequence `(B, 1568, 384)` → global mean pool → `z_tgt` `(B, 384)`.
* [cite_start]**Prediction Strategy**: The predictor operates on the *full token sequence* to preserve spatial and temporal structure during prediction, allowing cross-token attention before summarizing into a single vector[cite: 14].
* [cite_start]**Loss**: **VICReg** loss computed between `z_pred` and `z_tgt` to prevent representation collapse without requiring contrastive pairs or momentum encoders[cite: 16].

**Evaluation (Predictor discarded):**
* clip → **Encoder** → global mean pool → `(B, 384)` → **Linear Probe / kNN**.

---

### **Components**

#### **1. PatchEmbed3D**
[cite_start]Converts a spatiotemporal clip into tokens using a single Conv3D layer[cite: 18].
| Property | Value |
| :--- | :--- |
| **Kernel / Stride** | `(tubelet=2, patch=16, patch=16)` |
| **Token grid** | 8T × 14H × 14W = 1568 tokens |
| **Output dim** | 384 |
| **Post-norm** | LayerNorm |

#### **2. Positional Embedding**
[cite_start]Fixed 3D sinusoidal positional embedding[cite: 18]. Splits the embedding dimension independently across the temporal, height, and width axes.

#### **3. ViT Encoder**
| Property | Value |
| :--- | :--- |
| **Embed dim** | 384 |
| **Depth** | 8 transformer blocks |
| **Attention heads** | 6 |
| **MLP ratio** | 4.0 |
| **Pooling** | Global mean pool over all tokens → `(B, 384)` (Target path & Eval) |

#### **4. Shallow Transformer Predictor**
[cite_start]A lightweight, 2-layer Transformer operating on the full sequence before projecting back to the encoder dimension[cite: 14].
| Property | Value |
| :--- | :--- |
| **Embed dim** | 192 (internal), projects back to 384 |
| **Depth** | 2 transformer blocks |
| **Attention heads** | 4 |

#### **5. VICReg Loss**
Applied between the predicted view `z_pred` and the target view `z_tgt`.
| Term | Weight | Purpose |
| :--- | :--- | :--- |
| **Invariance** | 2.0 | MSE(`z_pred`, `z_tgt`) — predict the future correctly |
| **Variance** | 40.0 | Keep per-dim std ≥ 1 — aggressively prevent collapse |
| **Covariance** | 2.0 | Decorrelate embedding dimensions |

> [cite_start]**Parameter Count**: The model is heavily optimized to remain strictly under the 100M parameter limit[cite: 17].

---

### **Dataset: active_matter**

Physical simulations of active matter fluid dynamics.
| Property | Value |
| :--- | :--- |
| **Source** | HuggingFace `polymathic-ai/active_matter` |
| **Input channels** | [cite_start]11 (concentration, velocity, D-tensor, E-tensor) [cite: 12] |
| **Spatial resolution** | 224×224 (random crop from 256×256) |
| **Temporal length** | 32 frames total (16 context + 16 target) |
| **Physical parameters** | alpha (active dipole strength), zeta (steric alignment) |

#### **Data Augmentation & Normalization (Training)**
* Random spatial crop to 224×224[cite: 19].
* Gaussian noise injection (`std=1.0`).
* Per-sample, per-channel z-score normalization across T×H×W[cite: 19].

---

### **Training Configuration**

| Hyperparameter | Value |
| :--- | :--- |
| **Epochs** | 100 |
| **Batch size (per GPU)** | 4 |
| **Effective batch size** | 64 (via gradient accumulation) |
| **Learning rate** | 1e-3 |
| **LR schedule** | Cosine annealing with linear warmup (5 epochs) [cite: 20] |
| **Weight decay** | 0.05 |
| **Optimizer** | AdamW |
| **Mixed precision** | bfloat16 (via `torch.amp`) |

---

### **Evaluation**

The frozen encoder (384-dim mean-pooled output) is evaluated by predicting the physical parameters alpha and zeta [cite: 21] using:
* [cite_start]**Linear probe** — single `nn.Linear(384, 1)` trained with MSE loss[cite: 21].
* [cite_start]**kNN regression** — k=20 nearest neighbors with cosine distance[cite: 21].

*Note: Both evaluated with MSE loss on z-score normalized targets.*

---

### **Files**

| File | Description |
| :--- | :--- |
| `model.py` | [cite_start]Core architecture: PatchEmbed3D, ViTEncoder, TransformerPredictor, VICRegLoss[cite: 18]. |
| `dataset.py` | [cite_start]HDF5 sliding window extraction, crops, and normalizations[cite: 19]. |
| `train.py` | [cite_start]Pipeline with accumulative gradients, cosine LR, WandB integration, and preemption handling[cite: 20]. |
| `evaluate.py` | [cite_start]Linear Probe + kNN evaluation suite[cite: 21]. |
| `train.sbatch` / `eval.sbatch` | [cite_start]Slurm scripts configured for NYU HPC with automatic preemption handlers[cite: 22]. |
| `smoke_test.py` | Validation script to verify model dimensions and backward pass integrity before GPU submission. |

---

### **Running**

**Training:**
```bash
python train.py

# Resume from checkpoint:
python train.py --resume /scratch/ok2287/checkpoints/vit_jepa/latest.pt
