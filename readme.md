# SpliceAI Head Interpretation with Sparse Autoencoders

This repository contains code for analyzing and interpreting the attention heads in SpliceAI using sparse autoencoders (SAEs).

## Overview

We implement several approaches to interpret the behavior of attention heads in SpliceAI:

- Vanilla SAE: Basic sparse autoencoder training on attention head activations
- TopK SAE: Sparse autoencoder with top-k sparsity constraint
- BatchTopK SAE: Sparse autoencoder with batch-wise top-k sparsity
- Jump SAE: Sparse autoencoder with jump-start initialization

## Setup

The code requires:
- Python 3.9+
- PyTorch
- Anaconda/Conda environment

Use the provided environment file to set up dependencies:

```bash
conda env create -f server_env.yml
conda activate architecture_search_env
```


## Training

To train the vanilla, top-k, batch-topk, and jump SAEs on the SpliceAI model, run the following commands:

```bash
sbatch ./training_scripts/SpliceAI_10k_vanilla_SAE_training_job.sh
sbatch ./training_scripts/SpliceAI_10k_topk_SAE_training_job.sh
sbatch ./training_scripts/SpliceAI_10k_batchtopk_SAE_training_job.sh
sbatch ./training_scripts/SpliceAI_10k_jump_SAE_training_job.sh
```

These will train the vanilla, top-k, batch-topk, and jump SAEs on SpliceAI and save the model weights and metrics to the `out` directory.


## customization

The code is designed to be easily customizable.

And many of the parameters are set in the arguments to the `Train_SpliceAI_SAE.py` file.
Including the SAE type, the dictionary size, the learning rate, the number of epochs, and the batch size.

## Using directly from a python script

The code can also be used directly from a python script.







