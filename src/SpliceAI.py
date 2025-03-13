"""
Implementation of the SpliceAI deep learning architecture for splice site prediction.

This module contains the core SpliceAI model components including residual blocks,
mega blocks (sequences of residual blocks), and the full SpliceAI architecture.

The architecture follows the design described in Jaganathan et al. 2019.
"""

import lightning.pytorch as L
import torch.nn as nn
import numpy as np
from typing import Union
import torch
import h5py
from functools import reduce

class ResidualBlock(nn.Module):
    """
    A residual block in the style of SpliceAI that applies two convolutional layers with batch normalization,
    ReLU activation, and optional channel dropout.

    The block performs: x -> BN -> ReLU -> (Dropout) -> Conv -> BN -> ReLU -> (Dropout) -> Conv + x

    References
    ----------
    .. [1] Jaganathan et al. "Predicting Splicing from Primary Sequence with Deep Learning", Cell 2019
    .. [2] Cai et al. "Effective and Efficient Dropout for Deep Convolutional Neural Networks", arXiv 2019
    """

    def __init__(self, kernels, window, dilation, dropout_rate=None):
        """
        Initialize the residual block.

        Parameters
        ----------
        kernels : int
            Number of convolutional filters
        window : int 
            Kernel width for the convolutions
        dilation : int
            Dilation rate for the convolutions
        dropout_rate : float or None, default=None
            If not None, applies channel dropout with this probability
        """
        super().__init__()
        if dropout_rate is None:
            self.block = nn.Sequential(nn.BatchNorm1d(kernels),
                                       nn.ReLU(),
                                       nn.Conv1d(kernels, kernels, window, dilation=dilation, padding='same'),
                                       nn.BatchNorm1d(kernels),
                                       nn.ReLU(),
                                       nn.Conv1d(kernels, kernels, window, dilation=dilation, padding='same'))
        else:
            self.block = nn.Sequential(nn.BatchNorm1d(kernels),
                                       nn.ReLU(),
                                       nn.Dropout1d(p=dropout_rate),
                                       nn.Conv1d(kernels, kernels, window, dilation=dilation, padding='same'),
                                       nn.BatchNorm1d(kernels),
                                       nn.ReLU(),
                                       nn.Dropout1d(p=dropout_rate),
                                       nn.Conv1d(kernels, kernels, window, dilation=dilation, padding='same'))
        
        
    def forward(self, x):
        """
        Apply the residual block to the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, kernels, sequence_length)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, kernels, sequence_length)
        """
        return(x+ self.block(x))

        
class MegaBlock(nn.Module):
    """
    A sequence of residual blocks with shared parameters (kernel size, dilation rate).
    Default is 4 residual blocks as in the original SpliceAI paper.
    """

    def __init__(self, kernels, window, dilation, dropout_rate=None, block_count=4):
        """
        Initialize the MegaBlock.

        Parameters
        ----------
        kernels : int
            Number of convolutional filters
        window : int
            Kernel width for all convolutions
        dilation : int 
            Dilation rate for all convolutions
        dropout_rate : float or None, default=None
            If not None, applies channel dropout with this probability
        block_count : int, default=4
            Number of residual blocks to use
        """
        super().__init__()

        self.megablock = nn.Sequential(*[ResidualBlock(kernels, window, 
                                                       dilation, dropout_rate=dropout_rate) for i in range(block_count)])

    def forward(self, x):
        """
        Apply the sequence of residual blocks.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, kernels, sequence_length)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, kernels, sequence_length)
        """
        return(self.megablock(x))

class SpliceAI(L.LightningModule):
    """
    The core SpliceAI architecture consisting of multiple MegaBlocks with skip connections.
    
    The network structure is:
    1. Project input sequence to higher dimensions
    2. Apply series of MegaBlocks with increasing dilation rates
    3. Combine MegaBlock outputs via skip connections
    4. Project to output dimensions

    Parameters
    ----------
    k : int
        Number of filters in convolutional layers
    wsets : list of int
        Kernel widths for each MegaBlock
    dsets : list of int
        Dilation rates for each MegaBlock
    nt_dims : int, default=4
        Number of input channels (e.g. 4 for one-hot encoded DNA)
    output_dim : int, default=3
        Number of output channels
    dropout_rate : float or None, default=None
        If not None, applies channel dropout with this probability
    block_count : int, default=4
        Number of residual blocks per MegaBlock
    embedding_layer : str, default='mb 1'
        Which layer to use for embeddings
    embedding_dim : int, default=32
        Dimension of embeddings
    positions_to_use : list, default=[0,75]
        Which positions to use for embeddings
    input_length : int, default=176
        Length of input sequences
    """

    def __init__(self, k, wsets, dsets, nt_dims=4, output_dim=3, dropout_rate=None, block_count=4, embedding_layer='mb 1', embedding_dim=32, positions_to_use=[0,75], input_length=176):
        super().__init__()
        
        wsets = np.array(wsets)
        dsets = np.array(dsets)
        
        self.dropout_rate = dropout_rate
        
        self.cl = block_count*2 * np.sum(dsets*(wsets-1))

        #project
        self.first_projection = nn.Conv1d(nt_dims, k, 1, dilation=1, padding='same')
        self.firstbypass = nn.Conv1d(k, k, 1, dilation=1, padding='same')

        #blocks
        self.megablocks = nn.ModuleList([MegaBlock(k, w, d, 
                                                   dropout_rate=self.dropout_rate, block_count=block_count) for w, d in zip(wsets, dsets)])
        self.bypasses = nn.ModuleList([nn.Conv1d(k, k, 1, dilation=1, padding='same') for i in range(len(wsets))])

        #final proj
        self.final_projection = nn.Conv1d(k, output_dim, 1)

        self.hook_store = {}
        self.positions_to_use = positions_to_use
        
        if isinstance(self.positions_to_use, int):
            n_positions = 1
        elif isinstance(self.positions_to_use, slice):
            n_positions = len(range(*self.positions_to_use.indices(input_length))) 
        elif isinstance(self.positions_to_use, (list, torch.Tensor)):
            n_positions = len(self.positions_to_use)

        self.expected_size = n_positions*embedding_dim

        if embedding_layer.startswith('mb'):
            self.embed = lambda x: self.get_main_stream(x, int(embedding_layer.split(' ')[1]))[:,:,self.positions_to_use].flatten(start_dim=1)
        elif embedding_layer.startswith('input') or embedding_layer.startswith('output'):
            self.set_hooks(embedding_layer)
            self.embed = lambda x: self.get_activations(x)[embedding_layer.split(' ')[1]][:,:,self.positions_to_use].flatten(start_dim=1)
        elif embedding_layer == 'prob':
            self.set_hooks('output final_projection')
            self.embed = lambda x: torch.softmax(self.get_activations(x)['final_projection'], dim=1)[:,:,self.positions_to_use].flatten(start_dim=1)
        else:
            raise ValueError(f'Invalid embedding layer: {embedding_layer}')

        

    def construct_state_dict(self, h5_file, add_extras=True):
        """
        Construct state dict from h5 file.

        Parameters
        ----------
        h5_file : str
            Path to h5 file containing model weights
        add_extras : bool, default=True
            Whether to add extra parameters

        Returns
        -------
        dict
            State dict containing model parameters
        """
        num_mb = len(self.megablocks)
        translation_dict = {}

        for i in range(0,num_mb*8):
            mb, wi_mb = divmod(i, 8)
            rb, wi_rb = divmod(wi_mb,2)
            name = f'megablocks.{mb}.megablock.{rb}.block.{[0,3][wi_rb]}'

            if add_extras:
                pref = f'model_weights/batch_normalization_{i+1}/'
                suf = ':0'
            else:
                pref=''
                suf=''

            tmp_dict = {
                pref+f'batch_normalization_{i+1}/beta'+suf:name+'.bias',
                pref+f'batch_normalization_{i+1}/gamma'+suf:name+'.weight',
                pref+f'batch_normalization_{i+1}/moving_mean'+suf:name+'.running_mean',
                pref+f'batch_normalization_{i+1}/moving_variance'+suf:name+'.running_var',
            }
            translation_dict.update(tmp_dict)

        for i in range(0,num_mb*9+2):
            if i == 0:
                name = 'first_projection'
            elif i == 1:
                name = 'firstbypass'
            else:
                mb, wi_mb = divmod(i-2, 9)
                if wi_mb == 8:
                    name = f'bypasses.{mb}'
                else:
                    rb, wi_rb = divmod(wi_mb,2)
                    name = f'megablocks.{mb}.megablock.{rb}.block.{[2,5][wi_rb]}'

            if add_extras:
                pref = f'model_weights/conv1d_{i+1}/'
                suf = ':0'
            else:
                pref=''
                suf=''

            tmp_dict = {
                pref+f'conv1d_{i+1}/bias'+suf:name+'.bias',
                pref+f'conv1d_{i+1}/kernel'+suf:name+'.weight',
            }
            translation_dict.update(tmp_dict)

        new_sd = {}
        with h5py.File(h5_file, 'r') as f:
            for tf_name, torch_name in translation_dict.items():
                OG_params = f[tf_name][:]
                if len(OG_params.shape)==1:
                    new_sd[torch_name] = torch.Tensor(OG_params)
                elif len(OG_params.shape)==3:
                    new_sd[torch_name] = torch.Tensor(OG_params.transpose(2,1,0))
                else:
                    raise ValueError(f'enexpected dim num: {tf_name=} {torch_name=} {OG_params.shape=}')


            if add_extras:
                pref = f'model_weights/conv1d_{num_mb*9+3}/'
                suf = ':0'
            else:
                pref=''
                suf=''

            new_sd['final_projection.weight'] = torch.Tensor(f[pref+f'conv1d_{num_mb*9+3}/kernel'+suf][:].transpose(2,1,0))
            new_sd['final_projection.bias'] = torch.Tensor(f[pref+f'conv1d_{num_mb*9+3}/bias'+suf][:])

            return(new_sd)

    def load_from_h5_file(self, h5_file, add_extras=True):
        """
        Load model weights from h5 file.

        Parameters
        ----------
        h5_file : str
            Path to h5 file containing model weights
        add_extras : bool, default=True
            Whether to add extra parameters

        Returns
        -------
        self
            Returns self after loading weights
        """
        state_dict = self.construct_state_dict(h5_file,add_extras=add_extras)
        return(self.load_state_dict(state_dict))
    
    def set_hooks(self, code_string):
        """
        Set hooks for getting activations.

        Parameters
        ----------
        code_string : str
            String specifying which activations to get
        """
        in_out, access_string = code_string.split()

        self.hook_store = {}
        def get_activation(name):
            if in_out=='input':
                def hook(model, input, output):
                        self.hook_store[name] = input.detach()
            elif in_out=='output':
                def hook(model, input, output):
                        self.hook_store[name] = output.detach()
            else:
                raise NotImplementedError('Set the mode on your code string')
            return(hook)
        
        reduce(getattr, access_string.split('.'), self).register_forward_hook(get_activation(access_string))

    def get_activations(self, x):
        """
        Get activations from hooks.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        dict
            Dictionary of activations
        """
        self.forward(x)
        return(self.hook_store)
    
    def get_main_stream(self, x, block=None):
        """
        Get activations from main stream.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        block : int or None, default=None
            Which block to get activations from

        Returns
        -------
        torch.Tensor
            Activations from main stream
        """
        # project
        x = self.first_projection(x)
        out = self.firstbypass(x)

        if block is None:
            block = len(self.megablocks)

        # run blocks
        for b, mb, bp in zip(range(block),self.megablocks, self.bypasses):
            x = mb(x)
            out = out + bp(x)

        return(out)
        

    def forward(self, x):
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, nt_dims, sequence_length)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim, trimmed_sequence_length)
        """
        # project
        x = self.first_projection(x)
        out = self.firstbypass(x)

        # run blocks
        for mb, bp in zip(self.megablocks, self.bypasses):
            x = mb(x)
            out = out + bp(x)

        # Process final embedding
        out = self.final_projection(out)
        out = out[:,:,self.cl//2:-self.cl//2]

        return(out)
    
    def predict_step(self, batch, batch_idx):
        """
        Prediction step.

        Parameters
        ----------
        batch : torch.Tensor
            Input batch
        batch_idx : int
            Batch index

        Returns
        -------
        torch.Tensor
            Model predictions
        """
        x = batch
        output = self.predict(x)
        return(output)
    
    @property
    def mode(self):
        """Get model mode."""
        return(self._mode)
    
    @mode.setter
    def mode(self, mode):
        """
        Set model mode.

        Parameters
        ----------
        mode : str
            Model mode ('embed' or 'full')
        """
        self._mode = mode

        if mode == 'embed':
            self.predict = lambda x: self.embed(x)
        elif mode == 'full':
            self.predict = lambda x: self.forward(x)
        else:
            raise ValueError(f'Invalid mode: {mode}')
