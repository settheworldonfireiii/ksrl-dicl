#!/bin/bash -l
#SBATCH --time=13:30:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=50g
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1

cd /scratch.global/lee02328/temp/ksrl-dicl

source /home/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh

#conda env remove -n dicl2 -y || true
#conda create -n dicl2 python=3.9 -y

conda activate dicl2

pip install .[rl]

dicl-sac --seed 42 \
    --env-id HalfCheetah-v4 \
    --total-timesteps 100000 \
    --exp_name "temp_remove" \
    --batch_size 128 \
    --llm_batch_size 7 \
    --llm_learning_frequency 256 \
    --context_length 300 \
    --interact_every 1000 \
    --learning_starts 5000 \
    --llm_learning_starts 10000 \
    --llm_model 'meta-llama/Llama-3.2-1B' \
    --method 'dicl_s_pca'