#!/bin/bash -l        
#SBATCH --time=1:00:00
#SBATCH --ntasks=1       # Single task is better for PyTorch 
#SBATCH --cpus-per-task=8  # Increase CPUs for data loading
#SBATCH --mem=100G
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH --job-name=diffusion_wandb



conda activate dicl

conda install mamba -n base -c conda-forge

mamba uninstall matplotlib
mamba install matplotlib=3.7.2