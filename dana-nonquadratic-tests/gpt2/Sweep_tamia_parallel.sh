#!/bin/bash
#SBATCH --account=aip-gidelgau
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:h100:1          # one GPU per task
#SBATCH --cpus-per-task=8
#SBATCH --mem=0                    # “alloc as needed” on Alliance
#SBATCH --array=0-3%4              # four tasks, max 4 running at once
#SBATCH --job-name=hpo-array
#SBATCH --output=logs/%x-%A_%a.out # %A job‑ID, %a task‑ID
#SBATCH --error=logs/%x-%A_%a.err # %A job‑ID, %a task‑ID

# ── Per‑task hyper‑parameters (adjust as you like) ───────────
case $SLURM_ARRAY_TASK_ID in
  0) p=0.0  ;;
  1) p=0.1  ;;
  2) p=0.2  ;;
  3) p=0.3  ;;
esac

export CLUSTER=tamia
module load arrow/18.1.0 
module load python/3.10.13
module load cuda/12.6
module load httpproxy
echo "Loaded all modules"
source ~/links/projects/aip-gidelgau/dferbach/my-envs/dana-env/bin/activate
echo "Activated dana-env"

python nanogpt_log_losses_dana.py --train_steps=100 --batch_size=32 --val_batch_size=32 --seq_len=32 --dana_g2=0.05 --dana_g3_iv=0.01 --dana_g3_p=-$p --weight_decay=0.01