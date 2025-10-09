#!/bin/bash -l
#SBATCH --time=13:30:00
#SBATCH --ntasks=4         # Increase number of tasks to match nproc_per_node
#SBATCH --cpus-per-task=4  # Allocate 4 CPUs per task for data loading workers
#SBATCH --mem=50g
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=radke149@umn.edu
#SBATCH -p interactive-gpu      
#SBATCH --gres=gpu:a40:2  # Request 2 GPUs to match nproc_per_node


cd /scratch.global/radke149/ksrl-dicl



#source /scratch.global/radke149/dicl-ksrl-py11
source /users/2/radke149/anaconda3/etc/profile.d/conda.sh

conda init
conda activate /scratch.global/radke149/dicl-ksrl-py11

module load cuda/12.1.1

export LD_LIBRARY_PATH="$(python - <<'PY'
import os, pathlib, nvidia.cudnn as cudnn
print(pathlib.Path(cudnn.__file__).parent/'lib')
PY
):${LD_LIBRARY_PATH}"
# keep your CUDA path
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/common/software/install/manual/cuda/12.1.1
python -c 'import jax; print(jax.devices())'



echo "Job started at: $(date)"

dicl-sac --seed $RANDOM --env-id HalfCheetah-v4 --total-timesteps 1000000 --exp_name "CERTAINEMENT_FINALLY_30SEP_LSQR_HALFCHEETAH_BASELINE_MINE_50p" --batch_size 128 --llm_batch_size 7 --llm_learning_frequency 256 --context_length 500 --interact_every 1 --learning_starts 5000 --llm_learning_starts 10000 --llm_model 'meta-llama/Llama-3.2-1B' --method 'dicl_s_pca'

echo "Job ended at: $(date)"

