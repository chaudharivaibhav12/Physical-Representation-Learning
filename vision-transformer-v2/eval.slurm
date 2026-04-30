#!/bin/bash
#SBATCH --job-name=vicreg_v4_eval
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=n1s8-t4-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/sb10583/logs/vicreg_v4_eval_%j.log
#SBATCH --requeue

CKPT_DIR=/scratch/sb10583/checkpoints/vicreg-v4
OUT_DIR=/scratch/sb10583/eval-results/vicreg-v4
mkdir -p $OUT_DIR

singularity exec --bind /scratch --nv \
  --overlay /scratch/sb10583/overlay-10GB-400K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate base
    cd /scratch/sb10583/Physical-Representation-Learning/vicreg-sarvesh-v4

    echo '=== latest.pt ===' | tee $OUT_DIR/latest_eval.txt
    python evaluate.py --checkpoint $CKPT_DIR/latest.pt --test 2>&1 | tee -a $OUT_DIR/latest_eval.txt
  "

echo "Eval complete. Results in $OUT_DIR"
