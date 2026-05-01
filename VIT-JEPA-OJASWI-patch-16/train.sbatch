#!/bin/bash
#SBATCH --job-name=vit_jepa
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/ok2287/logs/vit_jepa_%j.log
#SBATCH --signal=SIGUSR1@90

# --- THE AUTO-REQUEUE TRAP ---
# This catches the preemption signal from Google Cloud
# and explicitly puts the job back in the Pending queue.
handle_preemption() {
    echo "Preemption signal received at $(date). Forcefully requeueing..."
    scontrol requeue $SLURM_JOB_ID
    exit 0
}
trap 'handle_preemption' SIGTERM SIGUSR1

mkdir -p /scratch/ok2287/logs

CKPT=/scratch/ok2287/checkpoints/vit_jepa/latest.pt
RESUME_FLAG=""
if [ -f "$CKPT" ]; then
    RESUME_FLAG="--resume $CKPT"
    echo "Resuming from $CKPT"
fi

# Note: Changed to :ro (read-only) to prevent file lock corruption during preemption
singularity exec --nv \
  --overlay /scratch/ok2287/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate physrep
    cd /scratch/ok2287/vit_jepa/Physical-Representation-Learning/ViT-JEPA/
    python -u train.py $RESUME_FLAG
  "