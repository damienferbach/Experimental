#! /bin/bash
#SBATCH --output=gpt2_sweep.out
#SBATCH --error=gpt2_sweep.err
#SBATCH --open-mode=truncate
#SBATCH --time=3:00:00
#SBATCH --mem=128GB
#SBATCH --gpus-per-node=h100:4
#SBATCH --account=aip-gidelgau

export CLUSTER=tamia
module load arrow/18.1.0 
module load python/3.10.13
module load cuda/12.6
module load httpproxy
echo "loaded modules"
source ~/links/projects/aip-gidelgau/dferbach/my-envs/dana-env/bin/activate
echo "activated env"

for p in 1.0
do
    echo "Running with g3p = $p"
    python nanogpt_log_losses_dana.py --train_steps=100 --batch_size=32 --val_batch_size=32 --seq_len=32 --dana_g2=0.05 --dana_g3_iv=0.01 --dana_g3_p=-$p --weight_decay=0.01
done
