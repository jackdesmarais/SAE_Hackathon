
ks=(4 8 16 32)
dict_sizes=(128 256 512 1024 2048)
topk_auxs=(4 8 16 32 128 256)

for k in ${ks[@]}; do
    for dict_size in ${dict_sizes[@]}; do
        for topk_aux in ${topk_auxs[@]}; do
            aux_penalty=$(echo "scale=4; 1/$k" | bc)
            # Take max of topk_aux and dict_size-k to ensure enough auxiliary features
            topk_aux=$(( topk_aux > (dict_size-k) ? topk_aux : (dict_size-k) ))
            sbatch batch_topK_hyperParam_sweep/training_scripts/training_job.sh ${k} ${dict_size} ${topk_aux} ${aux_penalty}
        done
    done
done
