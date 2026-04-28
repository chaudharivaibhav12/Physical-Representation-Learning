#!/bin/bash
#SBATCH --job-name=vit_jepa_eval
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/sb10583/logs/vit_jepa_eval_%j.log
#SBATCH --requeue

CKPT_DIR=/scratch/sb10583/checkpoints/vit-jepa-sarvesh
OUT_DIR=/scratch/sb10583/eval-results/vit-jepa-sarvesh
mkdir -p $OUT_DIR

singularity exec --bind /scratch --nv \
  --overlay /scratch/sb10583/overlay-10GB-400K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate base
    cd /scratch/sb10583/Physical-Representation-Learning/vit-sarvesh

    echo '=== best.pt (valid) ===' | tee $OUT_DIR/best_eval.txt
    python eval.py --checkpoint $CKPT_DIR/best.pt --split valid 2>&1 | tee -a $OUT_DIR/best_eval.txt

    echo '=== best.pt (test) ===' | tee $OUT_DIR/best_eval_test.txt
    python eval.py --checkpoint $CKPT_DIR/best.pt --split test 2>&1 | tee -a $OUT_DIR/best_eval_test.txt

    echo '=== latest.pt (valid) ===' | tee $OUT_DIR/latest_eval.txt
    python eval.py --checkpoint $CKPT_DIR/latest.pt --split valid 2>&1 | tee -a $OUT_DIR/latest_eval.txt

    echo '=== latest.pt (test) ===' | tee $OUT_DIR/latest_eval_test.txt
    python eval.py --checkpoint $CKPT_DIR/latest.pt --split test 2>&1 | tee -a $OUT_DIR/latest_eval_test.txt
  "

echo "Eval complete. Results in $OUT_DIR"
