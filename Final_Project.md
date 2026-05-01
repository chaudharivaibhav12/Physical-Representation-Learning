# Final Project: Self-Supervised Representation Learning on Physical Simulations

## 1. Dataset Description

The dataset for this project is the **`active_matter`** physical simulation dataset. Each sample represents a simulation of active matter dynamics captured as a spatiotemporal field with a resolution of **224×224**, and **11 physical channels**.

> **Note:** The raw simulations are 81 time-steps of 256×256, but they are processed into 224×224 windows for training.

### Fields
The 11 channels comprise:
- A **concentration scalar field**
- A **velocity vector field**
- An **orientation tensor field**
- A **strain-rate tensor field**

### Splits
| Split | Samples |
|-------|---------|
| Training | 8,750 |
| Validation | 1,200 |
| Test | 1,300 |

### Physical Parameters (Labels)
Each simulation trajectory is governed by two underlying physical constants:
- **α** (active dipole strength) — 5 discrete values
- **ζ** (steric alignment) — 9 discrete values
- **45 unique parameter combinations** in total

### A Helpful Perspective
You can think of this task as being highly analogous to **video representation learning**. Just as a video is a sequence of 2D images (frames) with 3 RGB channels evolving over time, the `active_matter` physical simulation is a sequence of spatial grids with 11 physical channels evolving over time. Because of this structural similarity, techniques and architectures designed for video understanding (like **Video JEPA** or **VideoMAE**) can be directly adapted to model these physical systems.

---

## 2. Project Resources

- **EB-JEPA Codebase:** An open-source library for learning representations from images and videos. You may use this as a reference or build upon it.
- **Baseline Paper & Code:** A recent paper providing a baseline approach and context for this type of predictive modeling on physical simulations.
  - Reference: [arXiv:2603.13227](https://arxiv.org/abs/2603.13227) | Baseline GitHub Repository

---

## 3. Data Access & Download

The dataset is hosted on HuggingFace. We will be using the `active_matter` dataset from the HuggingFace repository **`polymathic-ai/active_matter`**.

### Downloading the Data
You can download the dataset using the HuggingFace CLI. First, ensure you have the CLI installed in your environment, then run the following command to download the dataset to your desired directory (e.g., `/scratch/$NETID/data`):

```bash
# HuggingFace CLI download command
huggingface-cli download polymathic-ai/active_matter --local-dir /scratch/$NETID/data
```

> **Note:** The dataset is approximately **52 GB**. Please ensure you have sufficient storage space in your `/scratch` directory before downloading.

---

## 4. Task Description

Your objective is to design and train a **representation learning model** to capture the temporal evolution of physical systems using the `active_matter` dataset.

### Specific Steps

**1. Representation Learning**
Train your model using a self-supervised or unsupervised learning objective (e.g., latent prediction, masked reconstruction, contrastive learning) to capture the underlying physical dynamics. You **must not** use the physical parameter labels (α and ζ) during this stage.

**2. Linear Probing & kNN Regression**
Freeze the encoder and evaluate the learned representations. You must evaluate your representations using both:
- A **single linear layer**
- **kNN regression**

...to predict the continuous values of α and ζ (normalized via z-score) using **Mean Squared Error (MSE) loss**.

> **Important:** You must treat this as a **regression task** using MSE loss. You are **not** allowed to treat it as a discrete classification task. Complex regression heads (e.g., MLPs or attentive pooling classifiers) and end-to-end finetuning of the backbone are **strictly prohibited** for your main evaluation.

---

## 5. Computing Resources

All students have access to HPC Cloud Bursting for this course.

### Access Method
- Access is provided exclusively through **Open OnDemand (OOD):** https://ood-burst-001.hpc.nyu.edu/
- **Note:** NYU VPN is required when working off-campus.

### Slurm Account & Quota
- **Account:** `csci_ga_2572-2026sp`
- **Quota:** 300 GPU hours (18,000 minutes) per student, with sufficient CPU time.

### Allowed Partitions
| Partition | Resources |
|-----------|-----------|
| `interactive`, `n2c48m24` | CPU only |
| `g2-standard-12` | 1 L4 GPU |
| `g2-standard-24` | 2 L4 GPUs |
| `g2-standard-48` | 4 L4 GPUs |
| `c12m85-a100-1` | 1 A100 40GB GPU |
| `c24m170-a100-2` | 2 A100 40GB GPUs |
| `n1s8-t4-1` | 1 T4 GPU |

### Spot Instance Policy (Important!)
Cloud resources run on **Google Cloud spot instances**, which may be preempted (shut down) at any time.

- You **must** enable checkpoint/restart for your training runs.
- Save your checkpoints to your `/scratch/$NETID` directory.
- Add the following directive to all your Slurm scripts so jobs are automatically requeued if instances are preempted:

```bash
#SBATCH --requeue
```

---

## 6. Grading

Your project will be evaluated based on the following criteria (**Total 100%**):

| Criteria | Description |
|----------|-------------|
| **Performance (Linear Probing & kNN)** | MSE of predicting normalized physical parameters (α and ζ) on val/test set using a frozen encoder. Both linear probing and kNN regression results must be reported. |
| **Scientific Rigor & Analysis** | Quality of ablation studies, architectural/methodological novelty, and comparison against an end-to-end supervised baseline. Clear justification of design choices. |
| **Reproducibility & Code Quality** | Clean, well-documented code. Proper use of experiment tracking (e.g., Weights & Biases), clear instructions for running your pipeline, and efficient use of compute resources. |
| **Report Quality** | Clarity, writing, figures/tables, related work positioning, limitations/ethics, and explicit statement of contributions. |

---

## 7. Rules and Restrictions

1. **Training from Scratch:** No pretrained weights allowed.
2. **Model Size Limit:** Fewer than 100 Million parameters total.
3. **Data Scope:** `active_matter` dataset only. No external data.
4. **No Training on Validation/Test Data:** Training split only for weight updates.
5. **Linear Probing and kNN Only:** Single linear layer and kNN for final evaluation. No MLPs or attention pooling.

---

## 8. Deliverables & Submission

- **Final Report (6–8 pages):** ICML 2026 Style Template. Must include "Statement of Contributions".
- **Code and Artifacts:** Reproducible training/eval code, configs, saved weights, logs, eval script, `ENV.md`, `requirements.txt`.

---

## 9. Reproducibility Checklist

- [ ] Fixed seeds and determinism flags; record seeds in logs.
- [ ] Exact data preprocessing and augmentations documented.
- [ ] Full configuration files checked in; command lines to reproduce results.
- [ ] Parameter count report (must be < 100M).
- [ ] Compute accounting: GPUs, hours, peak memory; mixed precision usage.
- [ ] Clear separation between representation learning and linear probing stages.
- [ ] No references to external data, models, or weights in code.

---

## 10. FAQ

1. **Can we use CLIP/DINO/VideoMAE pretrained weights?** No. Training from scratch only.
2. **Can we use external unlabeled data?** No.
3. **Can we fine-tune the entire backbone?** No. Frozen encoder + linear layer/kNN only for main eval.
4. **Are we required to train an end-to-end supervised baseline?** Not mandatory, but recommended for comparison.
5. **How much does final performance affect our grade?** Strong performance is rewarded, but methodological novelty and rigorous analysis are equally valued.
