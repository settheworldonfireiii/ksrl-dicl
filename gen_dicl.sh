#!/bin/bash -l
#SBATCH --time=4:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=50g
#SBATCH --tmp=30g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1

ENV_PATH="/scratch.global/lee02328/dicl2" #Where you want your env made

cd /scratch.global/lee02328/temp/ksrl-dicl #should be where you cloned the env

source /home/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh

conda activate "${ENV_PATH}"

conda install -c conda-forge tensorflow=2.17 -y

# --- Set XLA_FLAGS to help TensorFlow find libdevice ---
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX"
# ---

echo "Upgrading pip..."
pip install --upgrade pip #just in case

echo "Installing KSRL_DICL package and its other dependencies from requirements.txt..."

pip install .[rl]

echo "Starting dicl-sac experiment..."
dicl-sac --seed 42 \
    --env-id HalfCheetah-v4 \
    --total-timesteps 100000 \
    --exp_name "testing_sample_no_train" \
    --batch_size 128 \
    --llm_batch_size 8 \
    --llm_learning_frequency 256 \
    --context_length 300 \
    --interact_every 1000 \
    --learning_starts 5000 \
    --llm_learning_starts 10000 \
    --llm_model 'meta-llama/Llama-3.2-1B' \
    --method 'dicl_s_pca' \
    --no-use_ksd_weighting

#    --no-use_ksd_pruning \
echo "Job finished."