#!/bin/bash -l
#SBATCH --time=13:00:00
#SBATCH --ntasks=2         # Increase number of tasks to match nproc_per_node
#SBATCH --cpus-per-task=4  # Allocate 4 CPUs per task for data loading workers
#SBATCH --mem=50g
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=radke149@umn.edu



cd /scratch.global/radke149/gpt2mla/dicl


source /users/2/radke149/anaconda3/etc/profile.d/conda.sh


conda activate /scratch.global/radke149/dicl

export OMP_NUM_THREADS=4  # You can adjust this value as needed


sac --seed $RANDOM --env-id "HalfCheetah" --total-timesteps 1000000 --exp_name "test_baseline" --interact_every 1000 --batch_size 128 --learning_starts 5000
