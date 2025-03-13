
if __name__ == '__main__':
    ############################################################
    ################### 1. General setup #######################
    ################### Mostly not model specific ##############
    ############################################################
    import argparse
    import torch
    from SAE_models import get_cfg, TopKSAE, VanillaSAE, JumpReLUSAE, BatchTopKSAE
    from SAE_training import SAETraining
    from torch.utils.data import DataLoader
    import numpy as np
    import json
    
    # SpliceAI specific imports
    from datasets import LiaoDatasetEmbedded
    from SpliceAI import SpliceAI


    parser = argparse.ArgumentParser()
    #General training parameters
    parser.add_argument('--seed', type=int, default=49, help='Random seed for reproducibility')
    parser.add_argument('--batch-size', type=int, default=4096, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--l1-coeff', type=float, default=0, help='L1 regularization coefficient')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 parameter for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.99, help='Beta2 parameter for Adam optimizer')
    parser.add_argument('--max-grad-norm', type=float, default=100000, help='Maximum gradient norm for clipping')
    parser.add_argument('--act-size', type=int, default=768, help='Size of activation vectors')
    parser.add_argument('--dict-size', type=int, default=12288, help='Size of the learned dictionary')
    parser.add_argument('--wandb-project', type=str, default='sparse_autoencoders', help='Weights & Biases project name')
    parser.add_argument('--input-unit-norm', type=bool, default=True, help='Whether to normalize input vectors to unit norm')
    parser.add_argument('--perf-log-freq', type=int, default=1000, help='Frequency of performance logging')
    parser.add_argument('--sae-type', type=str, default='topk', help='Type of sparse autoencoder (topk, vanilla, jumprelu, or batch_topk)')
    parser.add_argument('--checkpoint-freq', type=int, default=10000, help='Frequency of model checkpointing')
    parser.add_argument('--n-batches-to-dead', type=int, default=5, help='Number of batches before considering a feature dead')
    parser.add_argument('--accelerator', type=str, default='auto', help='Accelerator type (cpu, gpu, etc)')
    parser.add_argument('--devices', type=str, default='auto', help='Device configuration')
    parser.add_argument('--include-checkpointing', type=bool, default=True, help='Whether to save model checkpoints')
    parser.add_argument('--include-early-stopping', type=bool, default=True, help='Whether to use early stopping')
    parser.add_argument('--track-LR', type=bool, default=True, help='Whether to track learning rate changes')
    parser.add_argument('--epochs', type=int, default=1000, help='maximum number of training epochs')
    parser.add_argument('--outpath', type=str, default='./out/', help='Output directory path')
    #Warmstart parameters
    parser.add_argument('--warmstart-batches', type=int, default=1000, help='Number of warmstart batches')
    parser.add_argument('--warmstart-start-factor', type=float, default=0.0001, help='Initial factor for warmstart')
    parser.add_argument('--warmstart-end-factor', type=float, default=1, help='Final factor for warmstart')
    #Learning rate scheduler parameters
    parser.add_argument('--scheduler', type=str, options=['none','RedOnPlateau', 'CosineAnnealing','OneCycleLR'], default='RedOnPlateau', help='Learning rate scheduler type')
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
    parser.add_argument('--top-k', type=int, default=32, help='Number of top activations to keep')
    parser.add_argument('--top-k-aux', type=int, default=512, help='Number of top activations for auxiliary loss')
    parser.add_argument('--aux-penalty', type=float, default=1/32, help='Penalty coefficient for auxiliary loss')

    # for jumprelu
    parser.add_argument('--bandwidth', type=float, default=0.001, help='Bandwidth parameter for JumpReLU activation')

    #SpliceAI specific
    parser.add_argument('--k', type=int, default=32, help='Number of filters in convolutional layers')
    parser.add_argument('--wsets', type=int, default=[11], nargs='+', help='Kernel widths for each MegaBlock')
    parser.add_argument('--dsets', type=int, default=[1], nargs='+', help='Dilation rates for each MegaBlock')
    parser.add_argument('--nt-dims', type=int, default=4, help='Number of nucleotide dimensions')
    parser.add_argument('--output-dim', type=int, default=3, help='Number of output dimensions')
    parser.add_argument('--dropout-rate', type=float, default=None, help='Dropout rate')
    parser.add_argument('--block-count', type=int, default=4, help='Number of MegaBlocks')
    parser.add_argument('--embedding-layer', type=str, default='mb 1', help='Embedding layer to use')
    parser.add_argument('--embedding-dim', type=int, default=32, help='Embedding dimension')
    parser.add_argument('--input-length', type=int, default=176, help='Input length')
    parser.add_argument('--positions-to-use', nargs='+', type=int, default=[0,75], help='Positions to use for training')
    parser.add_argument('--csv-path', type=str, default='./data/Liao_Dataset/liao_training_set.csv', help='Path to training set CSV file')
    parser.add_argument('--plasmid-path', type=str, default='./data/Liao_Dataset/liao_plasmid.gbk', help='Path to plasmid file')
    parser.add_argument('--flanking-len', type=int, default=6, help='Flanking length')
    parser.add_argument('--add-context-len', type=int, default=0, help='are the positions relative to the context length or the input length')
    parser.add_argument('--auto-find-bc-pos', type=bool, default=True, help='Whether to automatically find BC positions')
    parser.add_argument('--auto-find-ex-pos', type=bool, default=True, help='Whether to automatically find exon positions')
    parser.add_argument('--preload', type=bool, default=True, help='Whether to preload data')
    parser.add_argument('--preload-embeddings', type=bool, default=True, help='Whether to preload embeddings')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--spliceai-batch-size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--h5-file', type=str, default='SpliceAI_Models/SpliceNet80_g1.h5', help='Path to SpliceAI model file')
    
    ############################################################
    ################### 2. Training setup ######################
    ################### Not model specific #####################
    ############################################################
    
    args = parser.parse_args()

    cfg = get_cfg(**vars(args))

    trainer = SAETraining(cfg)


    ############################################################
    ################### 3. Data setup ##########################
    ################### Model specific #########################
    ############################################################

    def one_hot_encode(x):
        """
        Convert DNA sequence to one-hot encoded tensor.
        
        Parameters
        ----------
        x : numpy.ndarray
            Input sequence array
            
        Returns
        -------
        torch.Tensor
            One-hot encoded tensor of shape (4, sequence_length)
        """
        var_ar = x[:,None] == np.array(['A','C','G', 'T'])
        var_ar = var_ar.T
        var_t = torch.Tensor(var_ar).float()
        return(var_t)

    
    spliceai_model = SpliceAI(k=cfg['k'], 
                     wsets=cfg['wsets'], 
                     dsets=cfg['dsets'], 
                     nt_dims=cfg['nt_dims'], 
                     output_dim=cfg['output_dim'], 
                     dropout_rate=cfg['dropout_rate'], 
                     block_count=cfg['block_count'], 
                     embedding_layer=cfg['embedding_layer'], 
                     embedding_dim=cfg['embedding_dim'], 
                     input_length=cfg['input_length'], 
                     positions_to_use=cfg['positions_to_use'])
    
    spliceai_model.load_from_h5_file(cfg['h5_file'])
    spliceai_model.mode = 'embed'

    if cfg['add_context_len']:
        cfg['positions_to_use'] = [pos+cfg['context_len']//2 for pos in cfg['positions_to_use']]

    
    transform_x = one_hot_encode

    full_train_ds = LiaoDatasetEmbedded(cfg['csv_path'], 
                             cfg['plasmid_path'], 
                             cfg['context_len']+cfg['flanking_len'], 
                             spliceai_model, 
                             cfg['auto_find_bc_pos'],
                             auto_find_ex_pos=cfg['auto_find_ex_pos'], 
                             batch_size=cfg['batch_size'], 
                             transform_x=transform_x, 
                             preload=cfg['preload'], 
                             preload_embeddings=cfg['preload_embeddings'], 
                             trainer=trainer.trainer, 
                             num_workers=cfg['num_workers'])
    full_train_ds.open()
    train_size = int(len(full_train_ds)*0.8)
    ids = np.random.permutation(len(full_train_ds))
    train_ds = torch.utils.data.dataset.Subset(full_train_ds, ids[:train_size])
    train_dl = torch.utils.data.dataloader.DataLoader(train_ds, batch_size=cfg['spliceai_batch_size'], num_workers=cfg['num_workers'])
    val_ds = torch.utils.data.dataset.Subset(full_train_ds, ids[train_size:])
    val_dl = torch.utils.data.dataloader.DataLoader(val_ds, batch_size=cfg['spliceai_batch_size'], num_workers=cfg['num_workers'])

    test_ds = LiaoDatasetEmbedded(cfg['test_csv_path'], 
                             cfg['plasmid_path'], 
                             cfg['context_len']+cfg['flanking_len'], 
                             spliceai_model, 
                             cfg['auto_find_bc_pos'],
                             auto_find_ex_pos=cfg['auto_find_ex_pos'], 
                             batch_size=cfg['batch_size'], 
                             transform_x=transform_x, 
                             preload=cfg['preload'], 
                             preload_embeddings=cfg['preload_embeddings'], 
                             trainer=trainer.trainer, 
                             num_workers=cfg['num_workers'])
    test_ds.open()
    test_dl = torch.utils.data.dataloader.DataLoader(test_ds, batch_size=cfg['spliceai_batch_size'], num_workers=cfg['num_workers'])
    

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

    ############################################################
    ################### 5. Training ############################
    ################### Not Model specific #####################
    ############################################################

    final_model = trainer.train(model, train_dl, val_dl)


    ############################################################
    ################### 6. testing/validation ##################
    ################### Not Model specific #####################
    ############################################################

    val_metrics = trainer.validate(val_dl)

    print(val_metrics)
    with open(cfg['outpath'] + 'val_metrics.json', 'w') as f:
        json.dump(val_metrics, f)

    test_metrics = trainer.test(test_dl)

    print(test_metrics)
    with open(cfg['outpath'] + 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f)

