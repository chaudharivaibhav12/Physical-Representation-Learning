#!/bin/bash
#SBATCH --job-name=vicreg_sarvesh_v1
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --output=/scratch/sb10583/logs/%j_%x.out
#SBATCH --error=/scratch/sb10583/logs/%j_%x.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --requeue
#SBATCH --mail-type=END
#SBATCH --mail-user=sb10583@nyu.edu

mkdir -p /scratch/sb10583/logs
mkdir -p /scratch/sb10583/checkpoints/vicreg

singularity exec --bind /scratch --nv \
  --overlay /scratch/sb10583/overlay-10GB-400K.ext3 \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/home/sb10583/.local/bin:\$PATH
conda activate base

cd /scratch/sb10583/Physical-Representation-Learning/vicreg-sarvesh-v1

CKPT=/scratch/sb10583/checkpoints/vicreg/latest.pt
if [ -f \"\$CKPT\" ]; then
  echo 'Resuming from checkpoint: '\$CKPT
  python3 train.py --resume \$CKPT
else
  echo 'Starting fresh training run'
  python3 train.py
fi
"
