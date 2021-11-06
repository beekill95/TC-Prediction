#!/bin/bash -i

#SBATCH -J tc_resnet_hyperband
#SBATCH -p dl
#SBATCH -o logs/tc_resnet_hyperband_%j.txt
#SBATCH -e logs/tc_resnet_hyperband_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qmnguyen@iu.edu
#SBATCH --nodes=1
#SBATCH --gpus-per-node v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=64G

# Activate conda environment with tensorflow-gpu
conda activate tc_prediction

# Start the same job on different nodes.
cd network_search
srun papermill resnet_hyperband_search.ipynb resnet_hyperband_search_output.ipynb -p project_start_time '2021Nov5_1127'
