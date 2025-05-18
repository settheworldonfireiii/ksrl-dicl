#!/bin/bash -l
#SBATCH --time=13:30:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=50g
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu

cd /scratch.global/lee02328/temp/ksrl-dicl
#not done: SBATCH -p a100-4
#not done: SBATCH --gres=gpu:a100:1

source /home/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh

# 1. Remove the environment if it exists, and proceed even if it doesn't (|| true)
conda env remove -n dicl2 -y || true

conda clean --all -y || true

pip cache purge || true

# 2. Create the environment, with -y to auto-confirm
conda create -n dicl2 python=3.9 -y

conda activate dicl2

pip install .
pip install .[rl]

# 3. Install TensorFlow using pip, as the error originated from a pip-installed package
pip install tensorflow

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