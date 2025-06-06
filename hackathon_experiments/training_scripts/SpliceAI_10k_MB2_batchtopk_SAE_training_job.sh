#!/bin/bash
#SBATCH --output=./logs/%x_qsub.sh.stout.%A.%a      # Output file
#SBATCH --error=./logs/%x_qsub.sh.sterr.%A.%a      # Error file
#SBATCH --time=1:59:59             # Max time for the job (max 2 hours for the fast queue or 12 hours for the default queue)
#SBATCH --mem=80G                   # Memory request
#SBATCH --ntasks=1                  # The number of parallel tasks in the job
#SBATCH --cpus-per-task=24           # Number of CPUs
#SBATCH --gpus-per-task=1           # Number of GPUs
#SBATCH --qos=default                  # fast, default (or blank), bio_ai
#SBATCH --partition=gpuq            # Queue to use
#SBATCH --array=1                   # Set this to 1-numCV for cross-validation

module load EBModules Anaconda3/2022.05
conda init bash
source activate architecture_search_env

python ./Train_SpliceAI_SAE.py --act-size 64 \
                            --wsets 11 11 21 41 \
                            --dsets 1 4 10 25 \
                            --hook-point "output megablocks.2.megablock.2.block" \
                            --h5-file SpliceAI_Models/SpliceNet10000_g1.h5 \
                            --dict-size 1024 \
                            --sae-type topk \
                            --wandb-project sparse_autoencoders \
                            --top-k 5 \
                            --top-k-aux 128 \
                            --aux-penalty 0.2 
                            