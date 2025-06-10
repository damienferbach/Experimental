#! /bin/bash
#SBATCH --output=gpt2_sweep_.out
#SBATCH --error=gpt2_sweep.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:80GB
#SBATCH --partition=unkillable

#TAMIA CLUSTER
module load arrow/11.0.8
module load python/3.10.13
source ../../dana-env/bin/activate


#MILA CLUSTER
#module load anaconda/3
#conda activate dana-env

for p in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    echo "Running with g3p = $p"
    python nanogpt_log_losses_dana.py --train_steps=50000 --batch_size=32 --val_batch_size=32 --seq_len=1024 --dana_g2=0.05 --dana_g3_iv=0.01 --dana_g3_p=-$p
done
