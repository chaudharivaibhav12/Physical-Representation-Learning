## **Vision Transformer — ViT-JEPA on Active Matter**

[cite_start]Self-supervised representation learning on physical simulations using a **Joint-Embedding Predictive Architecture (JEPA)**[cite: 9]. [cite_start]The model is trained to predict the latent representations of future states from current ones using a self-supervised **VICReg** loss[cite: 10].

---

### **Architecture Overview**

**Training Pipeline:**
* [cite_start]**Encoder:** 3D ViT with spatiotemporal patch embedding[cite: 13].
* [cite_start]**Predictor:** A shallow 2-layer Transformer that operates on the full token sequence to preserve spatial and temporal structure during prediction[cite: 14].
* [cite_start]**Loss:** **VICReg** (Variance-Invariance-Covariance Regularization) to prevent representation collapse without requiring contrastive pairs or momentum encoders[cite: 16].
* [cite_start]**Parameters:** Optimized to stay under the 100M parameter limit[cite: 17].

---

### **Components**

#### **1. PatchEmbed3D**
Converts a spatiotemporal clip into tokens using a single Conv3D layer.
| Property | Value |
| :--- | :--- |
| **Patch Size** | [cite_start]32x32 (spatial) with a tubelet size of 2 (temporal) [cite: 15] |
| **Token grid** | 8T × 7H × 7W = 392 tokens |
| **Output dim** | 384 |
| **Post-norm** | LayerNorm |

#### **2. Positional Embedding**
Fixed 3D sinusoidal positional embedding. Splits the embedding dimension independently across the temporal, height, and width axes.

#### **3. ViT Encoder**
| Property | Value |
| :--- | :--- |
| **Embed dim** | 384 |
| **Depth** | 8 transformer blocks |
| **Attention heads** | 6 |
| **MLP ratio** | 4.0 |

#### **4. Shallow Transformer Predictor**
[cite_start]A lightweight, 2-layer Transformer operating on the full sequence before projecting back to the encoder dimension[cite: 14].
| Property | Value |
| :--- | :--- |
| **Embed dim** | 192 (internal), projects back to 384 |
| **Depth** | [cite_start]2 transformer blocks [cite: 14] |
| **Attention heads** | 4 |

#### **5. VICReg Loss**
| Term | Weight | Purpose |
| :--- | :--- | :--- |
| **Invariance** | 2.0 | MSE between predicted and target embeddings. |
| **Variance** | 40.0 | [cite_start]Relu-based variance constraint to ensure embedding diversity[cite: 25]. |
| **Covariance** | 2.0 | Off-diagonal 1/D scaling to decorrelate embedding dimensions. |

---

### **Dataset: active_matter**

| Property | Value |
| :--- | :--- |
| **Input channels** | [cite_start]11 physical channels—including concentration, velocity, D-tensor, and E-tensor [cite: 12] |
| **Spatial resolution** | 224×224 (random crop from 256×256) |
| **Temporal length** | 32 frames total (16 context + 16 target) |
| **Physical parameters** | alpha (active dipole strength), zeta (steric alignment) |

---

### **Training Configuration**

| Hyperparameter | Value |
| :--- | :--- |
| **Epochs** | 100 |
| **Batch size (per GPU)** | 4 |
| **Target batch size** | 64 |
| **Learning rate** | 1e-3 |
| **Hardware** | [cite_start]Optimized for an NVIDIA A100 but includes gradient accumulation to allow training on smaller GPUs [cite: 27] |
| **Health Check** | [cite_start]Embedding Std provides a real-time health check for representation collapse [cite: 26] |

---

### **Files**

| File | Description |
| :--- | :--- |
| `model.py` | [cite_start]Core architecture including the 3D Patch Embedding, 3D Sinusoidal Positional Embeddings, ViT Encoder, and Transformer Predictor[cite: 18]. |
| `dataset.py` | [cite_start]HDF5 data loader handling sliding window extraction (32 frames), random spatial cropping (224x224), and z-score normalization[cite: 19]. |
| `train.py` | [cite_start]Training pipeline featuring gradient accumulation, cosine LR scheduling with linear warmup, and WandB integration[cite: 20]. |
| `evaluate.py`| [cite_start]Evaluation suite using Linear Probing (single layer) and kNN Regression to predict physical parameters (alpha and zeta) from frozen embeddings[cite: 21]. |
| `*.sbatch` | [cite_start]Slurm scripts configured for HPC execution, including automated preemption handling for Google Cloud/SLURM environments[cite: 22]. |

---

### **Running**

**Training:**
```bash
# Fresh training
python train.py

# Resume from latest checkpoint
python train.py --resume /scratch/ok2287/checkpoints/vit_jepa/latest.pt
