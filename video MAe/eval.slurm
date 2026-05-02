#!/bin/bash
#SBATCH --job-name=videomae_v1_eval
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=n1s8-t4-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/sb10583/logs/videomae_v1_eval_%j.log

CKPT_DIR=/scratch/sb10583/checkpoints/videomae-v1
OUT_DIR=/scratch/sb10583/eval-results/videomae-v1
mkdir -p $OUT_DIR

# Use best.pt if available, else latest.pt
if [ -f "$CKPT_DIR/best.pt" ]; then
    CKPT=$CKPT_DIR/best.pt
    echo "Using best.pt"
else
    CKPT=$CKPT_DIR/latest.pt
    echo "best.pt not found — using latest.pt"
fi

singularity exec --bind /scratch --nv \
  --overlay /scratch/sb10583/overlay-10GB-400K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate base
    cd /scratch/sb10583/Physical-Representation-Learning/video\ MAe

    python evaluate.py \
      --checkpoint $CKPT \
      --data-dir /scratch/sb10583/data/data \
      --batch-size 32 \
      --stride 2 \
    2>&1 | tee $OUT_DIR/eval_results.txt
  "

echo "Eval complete. Results in $OUT_DIR/eval_results.txt"
