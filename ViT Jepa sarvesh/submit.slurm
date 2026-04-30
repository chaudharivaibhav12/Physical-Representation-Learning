#!/bin/bash
#SBATCH --job-name=vit_jepa_sarvesh
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/sb10583/logs/vit_jepa_sarvesh_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sb10583@nyu.edu
#SBATCH --signal=SIGUSR1@90

handle_preemption() {
    echo "preemption signal at $(date) — forwarding to Python, then requeueing $SLURM_JOB_ID"
    kill -SIGUSR1 $CHILD_PID 2>/dev/null
    wait $CHILD_PID 2>/dev/null
    scontrol requeue $SLURM_JOB_ID
    exit 0
}
trap 'handle_preemption' SIGTERM SIGUSR1

mkdir -p /scratch/sb10583/logs

CKPT=/scratch/sb10583/checkpoints/vit-jepa-sarvesh/latest.pt
RESUME_FLAG=""
if [ -f "$CKPT" ]; then
    RESUME_FLAG="--resume $CKPT"
    echo "resuming from $CKPT"
fi

singularity exec --bind /scratch --nv \
  --overlay /scratch/sb10583/overlay-10GB-400K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate base
    cd /scratch/sb10583/Physical-Representation-Learning/vit-sarvesh
    python -u train.py $RESUME_FLAG
  " &
CHILD_PID=$!
wait $CHILD_PID
