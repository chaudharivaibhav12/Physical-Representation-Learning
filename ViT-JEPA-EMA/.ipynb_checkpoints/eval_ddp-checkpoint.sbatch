#!/bin/bash
#SBATCH --job-name=ijepa_eval
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --output=/scratch/ok2287/logs/eval_ijepa_%j.log

mkdir -p /scratch/ok2287/logs
mkdir -p /scratch/ok2287/checkpoints/ijepa/embeddings

echo "Starting evaluation at $(date)"

# ── Preemption handler ────────────────────────────────────────────────
# If preempted, the embedding cache already saved whatever was extracted.
# On requeue, extraction resumes from where it left off (per-split caching).
EVAL_PID=""

handle_preemption() {
    echo "=================================================="
    echo "PREEMPTION at $(date) — embeddings already cached!"
    echo "Job will requeue and resume from cached embeddings."
    echo "=================================================="
    if [ -n "$EVAL_PID" ]; then
        kill -SIGTERM $EVAL_PID 2>/dev/null
        sleep 10
    fi
    exit 0
}

trap 'handle_preemption' SIGUSR1 SIGTERM

singularity exec --nv \
  --overlay /scratch/ok2287/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate physrep
    cd /scratch/ok2287/vit_jepa/Physical-Representation-Learning/ViT-JEPA-EMA/

    python evaluate_ddp.py \
        --checkpoint /scratch/ok2287/checkpoints/ijepa/latest.pt \
        --save
  " &

EVAL_PID=$!
echo "Eval launched with PID: $EVAL_PID"
wait $EVAL_PID
EXIT_CODE=$?
echo "Eval finished with exit code: $EXIT_CODE at $(date)"
exit $EXIT_CODE
