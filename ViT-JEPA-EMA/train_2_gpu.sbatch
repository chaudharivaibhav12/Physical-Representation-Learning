#!/bin/bash
#SBATCH --job-name=ijepa
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --output=/scratch/ok2287/logs/ijepa_%j.log

mkdir -p /scratch/ok2287/logs
mkdir -p /scratch/ok2287/checkpoints/ijepa

CKPT=/scratch/ok2287/checkpoints/ijepa/latest.pt
RESUME_FLAG=""
if [ -f "$CKPT" ]; then
    RESUME_FLAG="--resume $CKPT"
    echo "Resuming from $CKPT"
else
    echo "Starting fresh training..."
fi

# ── Preemption handler ────────────────────────────────────────────────
# B: prefix means signal is sent to the batch script process itself,
# not the job steps. This ensures the signal reaches us even inside
# Singularity. We forward it to the torchrun process group.
TORCHRUN_PID=""

forward_signal() {
    echo "======================================================"
    echo "PREEMPTION SIGNAL received at $(date)"
    echo "Forwarding SIGUSR1 to torchrun (PID: $TORCHRUN_PID)..."
    echo "======================================================"
    if [ -n "$TORCHRUN_PID" ]; then
        # Send to entire process group so all ranks get it
        kill -SIGUSR1 -$TORCHRUN_PID 2>/dev/null || kill -SIGUSR1 $TORCHRUN_PID 2>/dev/null
        # Give Python time to save checkpoint atomically
        echo "Waiting 60s for atomic checkpoint save..."
        sleep 60
    fi
    echo "Exiting cleanly — job will be requeued by Slurm."
    exit 0
}

trap 'forward_signal' SIGUSR1 SIGTERM

# ── Launch training inside Singularity ───────────────────────────────
singularity exec --nv \
  --overlay /scratch/ok2287/overlay-15GB-500K.ext3:rw \
  /share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate physrep
    cd /scratch/ok2287/vit_jepa/Physical-Representation-Learning/ViT-JEPA-EMA/

    torchrun \
        --nproc_per_node=2 \
        --master_port=29500 \
        train.py $RESUME_FLAG
  " &

# Store the PID of the singularity process
TORCHRUN_PID=$!
echo "Singularity/torchrun launched with PID: $TORCHRUN_PID"

# Wait for the process to finish
wait $TORCHRUN_PID
EXIT_CODE=$?

echo "Process finished with exit code: $EXIT_CODE"
exit $EXIT_CODE
