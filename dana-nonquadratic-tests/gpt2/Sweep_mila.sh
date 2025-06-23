#! /bin/bash
#SBATCH --output=gpt2_sweep_%j.out
#SBATCH --error=gpt2_sweep_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:80GB
#SBATCH --partition=long

export CLUSTER=mila
module load anaconda/3
conda activate dana-env
# for p in 5.0
# do
#     echo "Running with g3p = $p"
python nanogpt_log_losses_dana.py --train_steps=100000 --batch_size=32 --val_batch_size=32 --seq_len=1024 --dana_g2=0.25 --dana_g3_iv=0.2 --dana_g3_p=-1.0 --weight_decay=0.0001 --wandb=True --optimizer="rmsprop" --learning_rate=0.0001 --beta_2=0.999 --bias_correction=True
# done
