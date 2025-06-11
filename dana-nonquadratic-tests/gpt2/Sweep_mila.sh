#! /bin/bash
#SBATCH --output=gpt2_sweep.out
#SBATCH --error=gpt2_sweep.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:80GB
#SBATCH --partition=unkillable

export CLUSTER=mila
module load anaconda/3
conda activate dana-env

for p in 5.0
do
    echo "Running with g3p = $p"
    python nanogpt_log_losses_dana.py --train_steps=10000 --batch_size=32 --val_batch_size=32 --seq_len=1024 --dana_g2=0.05 --dana_g3_iv=0.01 --dana_g3_p=-$p --weight_decay=0.0 #--wandb="yes" --path=""
done
