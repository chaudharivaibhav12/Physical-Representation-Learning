#!/bin/bash
#SBATCH --job-name=vicreg_sarvesh_v1
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/logs/vicreg_%j.out
#SBATCH --error=/scratch/%u/logs/vicreg_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=sb10583@nyu.edu

mkdir -p /scratch/sb10583/logs
mkdir -p /scratch/sb10583/checkpoints/vicreg

singularity exec --bind /scratch --nv --overlay /scratch/sb10583/overlay-10GB-400K.ext3:rw \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c '
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate base

    CKPT=/scratch/sb10583/checkpoints/vicreg/latest.pt
    RESUME_FLAG=""
    if [ -f "$CKPT" ]; then
        RESUME_FLAG="--resume $CKPT"
        echo "Resuming from $CKPT"
    fi

    cd /scratch/sb10583/Physical-Representation-Learning/vicreg-sarvesh-v1
    python -u train.py $RESUME_FLAG
'
