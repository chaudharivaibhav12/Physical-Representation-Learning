# ViT-JEPA for Active Matter Physical Representations

This repository implements a **Joint-Embedding Predictive Architecture (JEPA)** using a **3D Vision Transformer (ViT)** to learn representations of active matter physical simulations. The model is trained to predict the latent representations of future states from current ones using a self-supervised **VICReg** loss.

## 🔬 Project Overview
The goal is to learn a robust physical representation of active matter fluid simulations. The dataset consists of 11 physical channels—including concentration, velocity, D-tensor, and E-tensor—across spatiotemporal clips.

### Key Architecture Details
* **Encoder:** 3D ViT with spatiotemporal patch embedding.
* **Predictor:** A shallow 2-layer Transformer that operates on the full token sequence to preserve spatial and temporal structure during prediction.
* **Patch Size:** 32x32 (spatial) with a tubelet size of 2 (temporal).
* **Loss:** **VICReg** (Variance-Invariance-Covariance Regularization) to prevent representation collapse without requiring contrastive pairs or momentum encoders.
* **Parameters:** Optimized to stay under the 100M parameter limit.

---

## 📁 Repository Structure
* **`model.py`**: Core architecture including the 3D Patch Embedding, 3D Sinusoidal Positional Embeddings, ViT Encoder, and Transformer Predictor.
* **`dataset.py`**: HDF5 data loader handling sliding window extraction (32 frames), random spatial cropping (224x224), and z-score normalization.
* **`train.py`**: Training pipeline featuring gradient accumulation, cosine LR scheduling with linear warmup, and WandB integration.
* **`evaluate.py`**: Evaluation suite using **Linear Probing** (single layer) and **kNN Regression** to predict physical parameters (alpha and zeta) from frozen embeddings.
* **`*.sbatch`**: Slurm scripts configured for HPC execution, including automated preemption handling for Google Cloud/SLURM environments.

---

## 🚀 Getting Started

### 1. Installation
Ensure you have the required environment:
```bash
conda activate physrep
pip install torch torchvision wandb h5py scikit-learn
```

### 2. Training
The model supports automatic checkpointing and resuming via `SIGUSR1` signal handling, making it resilient to preemption on HPC clusters.
```bash
# Fresh training
python train.py

# Resume from latest checkpoint
python train.py --resume /scratch/ok2287/checkpoints/vit_jepa/latest.pt
```

### 3. Evaluation
Evaluates the frozen encoder on the downstream task of predicting simulation parameters using normalized MSE:
```bash
python evaluate.py --checkpoint /path/to/best.pt --save
```

---

## 📊 Performance Monitoring
The training script logs the following metrics to **Weights & Biases**:
* **Invariance Loss**: MSE between predicted and target embeddings.
* **Variance Loss**: Relu-based variance constraint to ensure embedding diversity.
* **Covariance Loss**: Off-diagonal 1/D scaling to decorrelate embedding dimensions.
* **Embedding Std**: A real-time health check for representation collapse.

## 🛠 Hardware Configuration
The default configuration is optimized for an **NVIDIA A100** but includes gradient accumulation (`target_batch: 64`) to allow training on smaller GPUs with a base `batch_size: 4`.