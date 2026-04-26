#!/bin/bash
#SBATCH --job-name=vit-jepa-v2-p32
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/vc2836/DL/logs/vit_jepa_v2_p32_%j.out
#SBATCH --error=/scratch/vc2836/DL/logs/vit_jepa_v2_p32_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vc2836@nyu.edu
#SBATCH --signal=SIGUSR1@90

handle_preemption() {
    echo "Preemption signal received at $(date). Requeueing job..."
    scontrol requeue $SLURM_JOB_ID
    exit 0
}
trap 'handle_preemption' SIGTERM SIGUSR1

mkdir -p /scratch/vc2836/DL/logs
mkdir -p /scratch/vc2836/DL/checkpoints/vit_jepa_v2

singularity exec --nv --overlay /scratch/vc2836/overlay-physrep.ext3:rw \
  /share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
  /bin/bash -c '
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate physrep

    cd /scratch/vc2836/DL/Physical-Representation-Learning/ViT-JEPA
    export PYTHONUNBUFFERED=1

    CKPT_DIR=/scratch/vc2836/DL/checkpoints/vit_jepa_v2
    RESUME_FLAG=""
    if [ -f "$CKPT_DIR/latest.pt" ]; then
        RESUME_FLAG="--resume $CKPT_DIR/latest.pt"
        echo "Resuming from $CKPT_DIR/latest.pt"
    else
        echo "Starting fresh training run"
    fi

    torchrun \
        --nproc_per_node=2 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:29500 \
        train.py \
        --data-dir      /scratch/vc2836/DL/data/active_matter/data \
        --out-dir       /scratch/vc2836/DL/checkpoints/vit_jepa_v2 \
        --epochs        100 \
        --batch-size    4 \
        --lr            1e-3 \
        --stride        1 \
        --run-name      vit-jepa-v2-p32 \
        --wandb-project active-matter-jepa-v2 \
        --wandb-entity  vaibhavnyu-new-york-university \
        $RESUME_FLAG

'
