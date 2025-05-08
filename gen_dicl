#!/bin/bash -l
#SBATCH --time=13:30:00
#SBATCH --ntasks=4         # Increase number of tasks to match nproc_per_node
#SBATCH --cpus-per-task=4  # Allocate 4 CPUs per task for data loading workers
#SBATCH --mem=50g
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=radke149@umn.edu


cd /users/2/radke149/budgeting_vllm


source /users/2/radke149/anaconda3/etc/profile.d/conda.sh


conda activate /scratch.global/radke149/dicl-ksrl


dicl-sac --seed 42 --env-id HalfCheetah-v4 --total-timesteps 100000 --exp_name "test_5p_dicl_s_pca_cl300__seed_42_shuffled" --batch_size 128 --llm_batch_size 7 --llm_learning_frequency 256 --context_length 300 --interact_every 1000 --learning_starts 5000 --llm_learning_starts 10000 --llm_model 'meta-llama/Llama-3.2-1B' --method 'dicl_s_pca'



