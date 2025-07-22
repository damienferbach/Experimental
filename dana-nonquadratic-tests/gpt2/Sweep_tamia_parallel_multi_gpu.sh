#!/bin/bash
#SBATCH --account=aip-gidelgau
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0                    # “alloc as needed” on Alliance
#SBATCH --job-name=multi_gpu_tamia
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

export NCCL_DEBUG=INFO
export CLUSTER=tamia
export WANDB_API_KEY=bece9f2099e3e85e0ae9922002616cf20bd26946
module load arrow/18.1.0
#module load python/3.10.13
module load python/3.11.5
module load cuda/12.6
module load httpproxy
echo "loaded modules"
source ~/links/projects/aip-gidelgau/dferbach/my-envs/dana-env/bin/activate
echo "activated env"

dana_g3_p=1.0
python nanogpt_log_losses_dana_multi_gpu.py --train_steps=100000 --batch_size=32 --val_batch_size=32 --seq_len=1024 --dana_g2=0.005 --dana_g3_iv=0.2 --dana_g3_p=-${dana_g3_p} --weight_decay=0.0 --wandb=True --optimizer="dana" --model="GPT2-large"
