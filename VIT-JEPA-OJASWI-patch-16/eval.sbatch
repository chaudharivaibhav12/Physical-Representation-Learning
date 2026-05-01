#!/bin/bash
#SBATCH --job-name=vit_eval
#SBATCH --account=csci_3033_109-2026sp
#SBATCH --partition=n2c48m24                # <-- CHANGED: Replaced the A100 partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
                                       # <-- DELETED: --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@90
#SBATCH --output=/scratch/ok2287/logs/vit_eval_%j.log

singularity exec --nv \
  --overlay /scratch/ok2287/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate physrep
    cd /scratch/ok2287/vit_jepa/Physical-Representation-Learning/ViT-JEPA/
    python evaluate.py \
        --checkpoint /scratch/ok2287/checkpoints/vit_jepa/latest.pt \
        --save
  "
