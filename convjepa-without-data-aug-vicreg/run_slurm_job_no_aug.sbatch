#!/bin/bash
#SBATCH --job-name=convjepa-no-aug
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/logs/convjepa_no_aug_%j.out
#SBATCH --error=/scratch/%u/logs/convjepa_no_aug_%j.err

mkdir -p /scratch/vc2836/DL/logs

# The whole training command runs inside singularity+conda activation
singularity exec --nv --overlay /scratch/vc2836/overlay-physrep-3.ext3:ro \
  /share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
  /bin/bash -c '
        source /ext3/miniconda3/etc/profile.d/conda.sh
        conda activate physrep
        
        cd /scratch/vc2836/DL/Physical-Representation-Learning/convjepa-without-data-aug-vicreg
        
        CKPT_DIR=/scratch/vc2836/DL/checkpoints/activematter-convjepa-direct
        RESUME_FLAG=""
        if [ -f "$CKPT_DIR/latest.pt" ]; then
            RESUME_FLAG="--resume $CKPT_DIR/latest.pt"
            echo "Resuming from $CKPT_DIR/latest.pt"
        fi

        python train.py --config config.yaml \
            $RESUME_FLAG \
            experiment_group="activematter-conv-no-aug" \
            run_name="activematter-convjepa-direct" \
            train.num_epochs=25 \
            train.batch_size=8 \
            train.num_workers=4 \
            train.target_global_batch_size=256 \
            train.lr_scheduler_warmup_epochs=1 \
            train.save_every=1 \
            train.report_every=10 \
            out_path=/scratch/vc2836/DL/checkpoints
    '
