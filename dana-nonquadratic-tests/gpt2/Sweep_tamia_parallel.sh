#!/bin/bash
#SBATCH --account=aip-gidelgau
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0                    # “alloc as needed” on Alliance
#SBATCH --job-name=hpo-array
#SBATCH --output=logs/%x-%j_%t.out # %j job‑ID, %t task‑ID
#SBATCH --error=logs/%x-%j_%t.err # %j job‑ID, %t task‑ID

export CLUSTER=tamia
module load arrow/18.1.0 
module load python/3.10.13
module load cuda/12.6
module load httpproxy
echo "loaded modules"
source ~/links/projects/aip-gidelgau/dferbach/my-envs/dana-env/bin/activate
echo "activated env"

# Launch four copies in parallel; each sees one GPU
srun --ntasks=4 --cpus-per-task=$SLURM_CPUS_PER_GPU \
     --gpus-per-task=h100:1 --gpu-bind=single:1 --exclusive \
     bash -c '
        i=$SLURM_LOCALID                 # 0..3
        case $i in
          0) p=0.0 ;;
          1) p=0.1 ;;
          2) p=0.2 ;;
          3) p=0.3 ;;
        esac
        python nanogpt_log_losses_dana.py --train_steps=100 --batch_size=32 --val_batch_size=32 --seq_len=32 --dana_g2=0.05 --dana_g3_iv=0.01 --dana_g3_p=-${p} --weight_decay=0.01
     '