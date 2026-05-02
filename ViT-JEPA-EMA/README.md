## **Vision Transformer — I-JEPA on Active Matter**

Self-supervised representation learning on physical simulations using the **Joint-Embedding Predictive Architecture (I-JEPA)**. The model learns by predicting the representations of masked "target" spatio-temporal blocks from visible "context" blocks, using a target encoder updated via **Exponential Moving Average (EMA)**.

---

### **Architecture Overview**

**Training Pipeline:**
* **Target Path**: A full 11-channel clip is passed through the **Target Encoder** (EMA of context encoder) to generate representations for all patches. Target patches are extracted using sampled indices.
* **Context Path**: The same clip is masked; only visible (unmasked) patches are passed through the **Context Encoder**.
* **Prediction**: A narrow **Predictor ViT** takes the context embeddings and learnable **Mask Tokens** (augmented with positional embeddings) to predict the missing target representations.
* **Loss**: Average **L2 (MSE) loss** between the predicted patch embeddings and the target encoder's output.

**Evaluation (Predictor discarded):**
* clip -> **Target Encoder** -> mean pool -> `(B, 384)` -> **Linear Probe / kNN**.
* *Note: The Target Encoder is used for evaluation as it produces richer representations by seeing the full input.*

---

### **Components**

#### **1. PatchEmbed (Spatio-temporal)**
Treats each frame as an independent 11-channel image before applying 2D projection and adding temporal/spatial embeddings.
| Property | Value |
| :--- | :--- |
| **Input channels** | 11 |
| **Patch size** | 16x16 |
| **Spatial Positional** | Learned `1 x N_patches x D` (shared across frames) |
| **Temporal Positional** | Learned `1 x T x D` (one per frame) |

#### **2. Multi-Block Mask Sampler**
Implements the multi-block masking strategy to ensure a non-trivial prediction task.
| Property | Value |
| :--- | :--- |
| **Target Blocks** | 4 blocks, scale (0.15, 0.2), aspect ratio (0.75, 1.5) |
| **Context Blocks** | 1 block, scale (0.85, 1.0) |
| **Strategy** | Temporal tube masking (same spatial mask across all frames) |

#### **3. Encoders & Predictor**
| Property | Context/Target Encoder (ViT-Small) | Predictor (Narrow ViT) |
| :--- | :--- | :--- |
| **Embed dim** | 384 | 192 |
| **Depth** | 12 | 6 |
| **Heads** | 6 | 6 |
| **MLP Ratio** | 4.0 | 4.0 |

---

### **Dataset: active_matter**

Physical simulations of active matter dynamics stored in HDF5 format.
| Property | Value |
| :--- | :--- |
| **Source** | HuggingFace `polymathic-ai/active_matter` |
| **Input Channels** | 11 (Concentration, Velocity, D tensor, E tensor) |
| **Resolution** | 224x224 (Cropped from 256x256) |
| **Temporal Length** | 32 frames total (16 context + 16 target) |
| **Parameters** | alpha (Active dipole strength), zeta (Steric alignment) |

---

### **Evaluation Results**

The frozen **Target Encoder** is evaluated on predicting alpha and zeta using MSE on z-score normalized targets.
| Method | Target | Train MSE | Val MSE | Test MSE |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Probe** | alpha | 0.6897 | 0.4394 | 0.4659 |
| **Linear Probe** | zeta | 0.5461 | 0.4438 | 0.4956 |
| **kNN (k=20)** | alpha | 0.4977 | 0.5027 | 0.4724 |
| **kNN (k=20)** | zeta | 0.4065 | 0.4590 | 0.6420 |

> **Collapse Check**: The model is reported as **HEALTHY** with an `avg_std` of 0.1785 and **0 dead dimensions**.

---

### **Running**

**HPC Execution (Slurm):**
The pipeline is designed to be preemption-safe on NYU HPC, caching embeddings every 50 batches to resume progress.
```bash
sbatch eval_ddp.sbatch

 

**Manual Evaluation:**
 
```bash
python evaluate_ddp.py --checkpoint /path/to/latest.pt --save
