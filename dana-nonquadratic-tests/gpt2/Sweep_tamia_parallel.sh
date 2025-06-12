#!/bin/bash
#SBATCH --account=aip-gidelgau
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0                    # “alloc as needed” on Alliance
#SBATCH --job-name=hpo-array

export CLUSTER=tamia
export WANDB_API_KEY=bece9f2099e3e85e0ae9922002616cf20bd26946
module load arrow/18.1.0 
module load python/3.10.13
module load cuda/12.6
module load httpproxy
echo "loaded modules"
source ~/links/projects/aip-gidelgau/dferbach/my-envs/dana-env/bin/activate
echo "activated env"

# MANUAL SWEEP
# Launch four copies in parallel; each sees one GPU
# srun --ntasks=4 --cpus-per-task=$SLURM_CPUS_PER_GPU \
#      --gpus-per-task=h100:1 --gpu-bind=single:1 --output=logs/%x-%j_%t.out --error=logs/%x-%j_%t.err --exclusive \
#      bash -c '
#         i=$SLURM_LOCALID                 # 0..3
#         case $i in
#           0) wd=1.0 ;;
#           1) wd=0.1 ;;
#           2) wd=0.01 ;;
#           3) wd=0.001 ;;
#         esac
#         python nanogpt_log_losses_dana.py --train_steps=100000 --batch_size=32 --val_batch_size=32 --seq_len=1024 --dana_g2=0.05 --dana_g3_iv=0.01 --dana_g3_p=-5.0 --weight_decay=${wd}
#      '

# WANDB SWEEP
# Launch sweep and capture the Sweep ID
SWEEP_ID=$(wandb sweep sweep_config.yaml 2>&1| grep -o 'wandb agent .*' | awk  '{print $NF}')
echo "Created sweep with ID: $SWEEP_ID"

# Check if sweep creation succeeded
if [ -z "$SWEEP_ID" ]; then
    echo $SWEEP_ID
    echo "Failed to create sweep or extract Sweep ID!"
    exit 1
fi

# Start 4 parallel agents (1 per GPU)
srun --ntasks=4 --cpus-per-task=$SLURM_CPUS_PER_GPU \
     --gpus-per-task=h100:1 --gpu-bind=single:1 --exclusive \
     bash -c "export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID && wandb agent $SWEEP_ID"
