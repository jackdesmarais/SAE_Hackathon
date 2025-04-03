#!/bin/bash
#SBATCH --output=./batch_topK_hyperParam_sweep/logs/V6/%x_qsub.sh.stout.%A.%a      # Output file
#SBATCH --error=./batch_topK_hyperParam_sweep/logs/V6/%x_qsub.sh.sterr.%A.%a      # Error file
#SBATCH --time=11:59:59             # Max time for the job (max 2 hours for the fast queue or 12 hours for the default queue)
#SBATCH --mem=250G                   # Memory request
#SBATCH --ntasks=1                  # The number of parallel tasks in the job
#SBATCH --cpus-per-task=24           # Number of CPUs
#SBATCH --gpus-per-task=1           # Number of GPUs
#SBATCH --qos=default                  # fast, default (or blank), bio_ai
#SBATCH --partition=gpuq            # Queue to use
#SBATCH --array=1                   # Set this to 1-numCV for cross-validation

module load EBModules Anaconda3/2022.05
conda init bash
source activate architecture_search_env

# Parse command line arguments
topk=$1
dict_size=$2
topk_aux=$3
aux_penalty=$4
workers=$5

python ./Train_SpliceAI_WG_SAE.py --act-size 32 \
                            --sae-type batch_topk \
                            --top-k ${topk} \
                            --dict-size ${dict_size} \
                            --batch-size 2097152 \
                            --n-batches-to-dead 5 \
                            --top-k-aux ${topk_aux} \
                            --aux-penalty ${aux_penalty} \
                            --exp-name "v6_testing_no_preload_batchtopk_HP_sweep" \
                            --overwite-name "SpliceAI_WG_Add_14_MB_3_out_topk${topk}_dict${dict_size}_topkaux${topk_aux}_auxpen${aux_penalty}" \
                            --outpath ./batch_topK_hyperParam_sweep/out/V6/ \
                            --num-workers ${workers} #\
                            # --preload-data

                            