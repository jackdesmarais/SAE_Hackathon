
ks=(16 24)
dict_sizes=(256 512)
topk_auxs=(16 32 128)
workers=64
aux_penalties=(1 0.5 0.1 0.01)
for k in ${ks[@]}; do
    for dict_size in ${dict_sizes[@]}; do
        for topk_aux in ${topk_auxs[@]}; do
            for aux_penalty in ${aux_penalties[@]}; do
                # Take min of topk_aux and dict_size-k to ensure enough auxiliary features
                topk_aux=$(( topk_aux < (dict_size-k) ? topk_aux : (dict_size-k) ))
                sbatch \
                    --job-name="v11_hp_sweep_k${k}_dict${dict_size}_topkaux${topk_aux}_auxpen${aux_penalty}" \
                    batch_topK_hyperParam_sweep/training_scripts/training_job.sh ${k} ${dict_size} ${topk_aux} ${aux_penalty} ${workers}
            done
        done
    done
done
