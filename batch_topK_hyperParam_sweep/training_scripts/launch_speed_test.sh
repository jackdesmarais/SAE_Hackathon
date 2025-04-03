
ks=(16)
dict_sizes=(1024)
topk_auxs=(16)
workers=(2 4 6 8 10 12 14 16)
for k in ${ks[@]}; do
    for dict_size in ${dict_sizes[@]}; do
        for topk_aux in ${topk_auxs[@]}; do
            for workers in ${workers[@]}; do
                aux_penalty=$(echo "scale=4; 1/$k" | bc)
                # Take min of topk_aux and dict_size-k to ensure enough auxiliary features
                topk_aux=$(( topk_aux < (dict_size-k) ? topk_aux : (dict_size-k) ))
                sbatch \
                --job-name="v5_w${workers}_testing_speed_batchtopk_HP_sweep_k${k}_dict${dict_size}_topkaux${topk_aux}_auxpen${aux_penalty}" \
                batch_topK_hyperParam_sweep/training_scripts/training_job.sh ${k} ${dict_size} ${topk_aux} ${aux_penalty} ${workers}
            done
        done
    done
done
