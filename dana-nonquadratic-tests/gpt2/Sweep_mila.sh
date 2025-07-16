#! /bin/bash
#SBATCH --output=gpt2_rope_%j.out
#SBATCH --error=gpt2_rope_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:80GB
#SBATCH --partition=main

export CLUSTER=mila
module load anaconda/3
conda activate dana-env
# for p in 5.0
# do
#     echo "Running with g3p = $p"
python ../../timescale-experiment/nanogpt_rmsprop_dana_baseline_mixed_bf16_rope.py --train_steps=100000 --batch_size=32 --val_batch_size=32 --seq_len=1024 --dana_g2=0.25 --dana_g3=0.05 --dana_kappa=1.0 --wandb=True
# done 
