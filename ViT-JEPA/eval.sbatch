#!/bin/bash
#SBATCH --job-name=vit_eval
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ok2287/logs/vit_eval_%j.log

singularity exec --nv \
  --overlay /scratch/ok2287/overlay-15GB-500K.ext3:rw \
  /share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate physrep
    cd /scratch/ok2287/vit_jepa/Physical-Representation-Learning/ViT-JEPA/
    python evaluate.py \
        --checkpoint /scratch/ok2287/checkpoints/vit_jepa/best.pt \
        --save
  "
