
if __name__ == '__main__':
    ############################################################
    ################### 1. General setup #######################
    ################### Mostly not model specific ##############
    ############################################################
    import sys
    sys.path.append('./src')
    import argparse
    import torch
    from SAE_models import get_cfg, TopKSAE, VanillaSAE, JumpReLUSAE, BatchTopKSAE
    from SAE_training import SAETraining
    from torch.utils.data import DataLoader
    import numpy as np
    import json
    
    # SpliceAI specific imports
    from hdf_dataset import HDF3DIterator
    from SpliceAI import SpliceAI


    parser = argparse.ArgumentParser()
    #General training parameters
    parser.add_argument('--seed', type=int, default=49, help='Random seed for reproducibility')
    parser.add_argument('--batch-size', type=int, default=16384, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--l1-coeff', type=float, default=0, help='L1 regularization coefficient')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 parameter for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.99, help='Beta2 parameter for Adam optimizer')
    parser.add_argument('--max-grad-norm', type=float, default=100000, help='Maximum gradient norm for clipping')
    parser.add_argument('--act-size', type=int, default=32, help='Size of activation vectors')
    parser.add_argument('--dict-size', type=int, default=1024, help='Size of the learned dictionary')
    parser.add_argument('--wandb-project', type=str, default='sparse_autoencoders', help='Weights & Biases project name')
    parser.add_argument('--exp-name', type=str, default='SpliceAI_WG_SAE', help='Name of the experiment')
    parser.add_argument('--overwite-name', type=str, default=None, help='Overwrite the experiment name')
    parser.add_argument('--input-unit-norm', action='store_true', help='Whether input embeddings are normalized to unit norm')
    parser.add_argument('--perf-log-freq', type=int, default=1000, help='Frequency of performance logging')
    parser.add_argument('--sae-type', type=str, default='topk', help='Type of sparse autoencoder (topk, vanilla, jumprelu, or batch_topk)')
    parser.add_argument('--checkpoint-freq', type=int, default=10000, help='Frequency of model checkpointing')
    parser.add_argument('--n-batches-to-dead', type=int, default=600, help='Number of batches before considering a feature dead')
    parser.add_argument('--accelerator', type=str, default='auto', help='Accelerator type (cpu, gpu, etc)')
    parser.add_argument('--devices', type=str, default='auto', help='Device configuration')
    parser.add_argument('--no-checkpointing', action='store_false', dest='include_checkpointing', help='Disable model checkpointing')
    parser.add_argument('--no-early-stopping', action='store_false', dest='include_early_stopping', help='Disable early stopping')
    parser.add_argument('--no-lr-tracking', action='store_false', dest='track_LR', help='Disable learning rate change tracking')
    parser.add_argument('--epochs', type=int, default=1000, help='maximum number of training epochs')
    parser.add_argument('--outpath', type=str, default='./out/', help='Output directory path')
    parser.add_argument('--num-workers', type=int, default=16, help='Number of workers for dataloader')
    #Warmstart parameters
    parser.add_argument('--warmstart-batches', type=int, default=0, help='Number of warmstart batches')
    parser.add_argument('--warmstart-start-factor', type=float, default=0.0001, help='Initial factor for warmstart')
    parser.add_argument('--warmstart-end-factor', type=float, default=1, help='Final factor for warmstart')
    #Learning rate scheduler parameters
    parser.add_argument('--scheduler', type=str, choices=['none','RedOnPlateau', 'CosineAnnealing','OneCycleLR'], default='RedOnPlateau', help='Learning rate scheduler type')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay for optimizer')
    #ReduceLROnPlateau parameters
    parser.add_argument('--reduceLROnPlateau-factor', type=float, default=0.1, help='Factor by which to reduce learning rate')
    parser.add_argument('--reduceLROnPlateau-patience', type=int, default=4, help='Patience for learning rate scheduler')
    parser.add_argument('--reduceLROnPlateau-threshold', type=float, default=0.0001, help='Threshold for learning rate scheduler')
    parser.add_argument('--reduceLROnPlateau-cooldown', type=int, default=0, help='Cooldown period for learning rate scheduler')
    parser.add_argument('--reduceLROnPlateau-min', type=float, default=0, help='Minimum learning rate')
    parser.add_argument('--reduceLROnPlateau-eps', type=float, default=1e-08, help='Epsilon for learning rate scheduler')
    #Early stopping parameters
    parser.add_argument('--min-delta', type=float, default=0, help='Minimum change in monitored quantity for early stopping')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')


    # (Batch)TopKSAE specific
    parser.add_argument('--top-k', type=int, default=16, help='Number of top activations to keep')
    parser.add_argument('--top-k-aux', type=int, default=256, help='Number of top activations for auxiliary loss')
    parser.add_argument('--aux-penalty', type=float, default=1/16, help='Penalty coefficient for auxiliary loss')

    # for jumprelu
    parser.add_argument('--bandwidth', type=float, default=0.001, help='Bandwidth parameter for JumpReLU activation')

    #SpliceAI specific
    parser.add_argument('--train-data-path', type=str, default='/grid/hackathon/data_norepl/splarseers/output/embed_train.h5', help='Path to SpliceAI embedding files')
    parser.add_argument('--val-data-path', type=str, default='/grid/hackathon/data_norepl/splarseers/output/embed_val.h5', help='Path to SpliceAI embedding files')
    parser.add_argument('--test-data-path', type=str, default='/grid/hackathon/data_norepl/splarseers/output/embed_test.h5', help='Path to SpliceAI embedding files')
    parser.add_argument('--preload-data', action='store_true', help='Preload data into memory')
    parser.add_argument('--train-dataset-name', type=str, default='embed_train', help='Name of the dataset')
    parser.add_argument('--val-dataset-name', type=str, default='embed_val', help='Name of the dataset')
    parser.add_argument('--test-dataset-name', type=str, default='embed_test', help='Name of the dataset')
    parser.add_argument('--hook-point', type=str, default='Add_14_MB_3_out', help='Hook point for SpliceAI model')
    parser.add_argument('--model-name', type=str, default='SpliceAI_WG', help='Name of the model')
    
    ############################################################
    ################### 2. Training setup ######################
    ################### Not model specific #####################
    ############################################################
    
    args = parser.parse_args()

    cfg = get_cfg(**vars(args))
    if args.overwite_name is not None:
        cfg['name'] = args.overwite_name
    print('cfg - set')
    print(cfg)


    trainer = SAETraining(cfg)
    print('trainer - set')


    ############################################################
    ################### 3. Data setup ##########################
    ################### Model specific #########################
    ############################################################

    
    train_ds = HDF3DIterator(cfg['train_data_path'], cfg['train_dataset_name'], preload=cfg['preload_data'])
    train_dl = torch.utils.data.dataloader.DataLoader(train_ds, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=True)
    val_ds = HDF3DIterator(cfg['val_data_path'], cfg['val_dataset_name'], preload=cfg['preload_data'])
    val_dl = torch.utils.data.dataloader.DataLoader(val_ds, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])

    test_ds = HDF3DIterator(cfg['test_data_path'], cfg['test_dataset_name'], preload=cfg['preload_data'])
    test_dl = torch.utils.data.dataloader.DataLoader(test_ds, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])
    print('data - set')

    ############################################################
    ################### 4. SAE Model setup #####################
    ################### Not Model specific #####################
    ############################################################

    cfg['training_set_batches'] = len(train_dl)


    if cfg['sae_type'] == 'topk':
        model = TopKSAE(cfg)
    elif cfg['sae_type'] == 'vanilla':
        model = VanillaSAE(cfg)
    elif cfg['sae_type'] == 'jumprelu':
        model = JumpReLUSAE(cfg)
    elif cfg['sae_type'] == 'batch_topk':
        model = BatchTopKSAE(cfg)
    print('model - set')

    ############################################################
    ################### 5. Training ############################
    ################### Not Model specific #####################
    ############################################################

    with train_ds, val_ds:
        print('training - starting')
        final_model = trainer.train(model, train_dl, val_dl)
    print('training - done')

    ############################################################
    ################### 6. testing/validation ##################
    ################### Not Model specific #####################
    ############################################################

    with val_ds:
        print('validation - starting')
        val_metrics = trainer.validate(val_dl)
    print('validation - done')

    print(val_metrics)
    with open(cfg['outpath'] + f"{cfg['name']}_{cfg['seed']}_val_metrics.json", 'w') as f:
        json.dump(val_metrics, f)

    with test_ds:
        print('testing - starting')
        test_metrics = trainer.test(test_dl)
    print('testing - done')

    print(test_metrics)
    with open(cfg['outpath'] + f"{cfg['name']}_{cfg['seed']}_test_metrics.json", 'w') as f:
        json.dump(test_metrics, f)

