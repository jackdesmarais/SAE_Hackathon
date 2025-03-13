#!/bin/bash
#SBATCH --output=./logs/%x_qsub.sh.stout.%A.%a      # Output file
#SBATCH --error=./logs/%x_qsub.sh.sterr.%A.%a      # Error file
#SBATCH --time=1:59:59             # Max time for the job (max 2 hours for the fast queue or 12 hours for the default queue)
#SBATCH --mem=200G                   # Memory request
#SBATCH --ntasks=1                  # The number of parallel tasks in the job
#SBATCH --cpus-per-task=24           # Number of CPUs
#SBATCH --gpus-per-task=1           # Number of GPUs
#SBATCH --qos=default                  # fast, default (or blank), bio_ai
#SBATCH --partition=gpuq            # Queue to use
#SBATCH --array=1                   # Set this to 1-numCV for cross-validation

module load EBModules Anaconda3/2022.05
conda init bash
source activate architecture_search_env

python ./Train_SpliceAI_WG_SAE.py --act-size 32 \
                            --top-k 8 \
                            --dict-size 1024 \
                            --batch-size 4194304 \
                            --n-batches-to-dead 3 \
                            --top-k-aux 256 \
                            --aux-penalty 0.125 \
                            --preload-data

                            