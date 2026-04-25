#!/bin/bash
#SBATCH --job-name=vicreg_sarvesh_v4
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/sb10583/logs/vicreg_%j.out
#SBATCH --error=/scratch/sb10583/logs/vicreg_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sb10583@nyu.edu
#SBATCH --signal=SIGUSR1@90

# --- AUTO-REQUEUE TRAP ---
# Catches preemption signal from Google Cloud spot instance
# and explicitly puts the job back in the queue
handle_preemption() {
    echo "Preemption signal received at $(date). Requeueing job..."
    scontrol requeue $SLURM_JOB_ID
    exit 0
}
trap 'handle_preemption' SIGTERM SIGUSR1

mkdir -p /scratch/sb10583/logs
mkdir -p /scratch/sb10583/checkpoints/vicreg-v4

CKPT=/scratch/sb10583/checkpoints/vicreg-v4/latest.pt
RESUME_FLAG=""
if [ -f "$CKPT" ]; then
    RESUME_FLAG="--resume $CKPT"
    echo "Resuming from $CKPT"
fi

singularity exec --bind /scratch --nv \
  --overlay /scratch/sb10583/overlay-10GB-400K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate base
    cd /scratch/sb10583/Physical-Representation-Learning/vicreg-sarvesh-v4
    python -u train.py $RESUME_FLAG
  "
