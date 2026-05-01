# I-JEPA for Active Matter Physical Representations

This repository implements an **Image-based Joint-Embedding Predictive Architecture (I-JEPA)** adapted for spatiotemporal physical simulations. Unlike contrastive methods, this architecture uses an **Exponential Moving Average (EMA)** target encoder and multi-block spatial masking to learn highly robust physical representations without requiring explicit collapse-prevention losses (like VICReg) or negative pairs.

## 🔬 Architecture Details
The architecture closely follows the original I-JEPA design but is extended for 11-channel active matter data (e.g., concentration, velocity, D-tensor, E-tensor).

* **Context Encoder:** A ViT-Small (12 layers, 6 heads, dim 384) that processes *only the visible* spatiotemporal patches.
* **Target Encoder:** An EMA copy of the Context Encoder. Momentum scales smoothly from 0.996 to 1.0 during training.
* **Predictor:** A narrow ViT bottleneck (6 layers, dim 192) that predicts the target patch representations using context tokens and positional mask tokens.
* **Masking Strategy:** Multi-block spatial masking (4 target blocks, 1 context block) extended into "temporal tubes" (the same spatial mask applied across all frames).
* **Data Configuration:** Subsampled to 4 frames per clip with a 32x32 spatial patch size, drastically reducing the token sequence and attention bottleneck.

---

## 📁 Repository Structure
* **`model.py`**: Core architecture including the 11-channel Spatiotemporal Patch Embedding, Context/Target ViTs, and the Predictor bottleneck.
* **`masking.py`**: Implements the multi-block spatial masking strategy defined in the I-JEPA paper.
* **`dataset.py`**: HDF5 data loader handling active matter channel concatenation, 224x224 random spatial crops, and independent channel z-score normalization.
* **`train.py`**: Main pretraining loop featuring cosine learning rate scheduling, EMA momentum scheduling, and atomic checkpointing.
* **`evaluate_ddp.py`**: Evaluation suite (Linear Probe and kNN Regression) with robust Slurm preemption handling via chunk-based embedding caching.
* **`*.sbatch`**: Slurm scripts tailored for single and multi-GPU (DDP) execution on HPC environments.

---

## 🚀 Getting Started

### 1. Installation
```bash
conda activate physrep
pip install torch torchvision wandb h5py scikit-learn
```

### 2. Pretraining
Training leverages atomic checkpointing to survive HPC preemption. 
```bash
# Single-GPU Training
sbatch train.sbatch

# Multi-GPU (DDP) Training
sbatch train_2_gpu.sbatch
```

### 3. Evaluation
Evaluates the *frozen Target Encoder* on downstream parameter prediction (alpha and zeta). The `evaluate_ddp.py` script automatically resumes from cached embeddings if preempted.
```bash
sbatch eval.sbatch

# or manually:
python evaluate_ddp.py --checkpoint /scratch/ok2287/checkpoints/ijepa/latest.pt --save
```

---

## 🛡 HPC / Slurm Resiliency
This codebase is heavily optimized for interruptible GPU clusters (e.g., Google Cloud/Slurm):
* **Atomic Checkpointing:** Writes to temporary files before renaming, eliminating corrupted `.pt` files.
* **Signal Handling:** Catches `SIGUSR1`/`SIGTERM` to gracefully pause and requeue.
* **Embedding Caching:** Evaluation extracts embeddings in chunks of 50 batches. If preempted, it skips completed chunks upon resuming.