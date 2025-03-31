"""
Sparse autoencoder models and utilities for training.

This module contains implementations of various sparse autoencoder architectures including:
- Base autoencoder class
- BatchTopK sparse autoencoder
- TopK sparse autoencoder 
- Vanilla sparse autoencoder
- JumpReLU sparse autoencoder

The implementations are adapted from https://github.com/bartbussmann/BatchTopK/blob/main/sae.py
"""

import torch 
import torch.nn as nn
import lightning.pytorch as L
from torch.optim.optimizer import Optimizer
import torchmetrics
import torch.nn.functional as F
import torch.autograd as autograd

def get_cfg(**kwargs):
    """
    Get default configuration dictionary with optional overrides.

    Args:
        **kwargs: Keyword arguments to override default config values

    Returns:
        dict: Configuration dictionary with default values and any overrides
    """
    default_cfg = {
        "seed": 49,
        "batch_size": 4096,
        "lr": 3e-4,
        "l1_coeff": 0,
        "beta1": 0.9,
        "beta2": 0.99,
        "max_grad_norm": 100000,
        "dtype": torch.float32,
        "act_size": 768,
        "dict_size": 12288,
        "wandb_project": "sparse_autoencoders",
        "input_unit_norm": True,
        "perf_log_freq": 1000,
        "sae_type": "topk",
        "checkpoint_freq": 10000,
        "n_batches_to_dead": 5,
        "warmstart_batches": 1000,
        "warmstart_start_factor": 0.0001,
        "warmstart_end_factor": 1,
        "scheduler": "RedOnPlateau",
        "weight_decay": 0.0001,
        "reduceLROnPlateau_factor": 0.1,
        "reduceLROnPlateau_patience": 4,
        "reduceLROnPlateau_threshold": 0.0001,
        "reduceLROnPlateau_cooldown": 0,
        "reduceLROnPlateau_min": 0,
        "reduceLROnPlateau_eps": 1e-08,
        "epochs": 1000,
        "training_set_batches": 1000,
        "outpath":"./out/",
        "min_delta":0,
        'patience':10,
        "accelerator":"auto",
        "devices":"auto",
        "include_checkpointing":True,
        "include_early_stopping":True,
        "track_LR":True,
        # (Batch)TopKSAE specific
        "top_k": 32,
        "top_k_aux": 512,
        "aux_penalty": (1/32),
        # for jumprelu
        "bandwidth": 0.001,
    }
    default_cfg.update(kwargs)
    default_cfg = post_init_cfg(default_cfg)
    return default_cfg

def post_init_cfg(cfg):
    """
    Post-process configuration dictionary to add derived fields.

    Args:
        cfg (dict): Configuration dictionary

    Returns:
        dict: Configuration with added name field
    """
    cfg["name"] = f"{cfg.get('model_name','UnknownModel')}_{cfg.get('hook_point','UnknownHookPoint')}_{cfg['dict_size']}_{cfg['sae_type']}_{cfg['top_k']}_{cfg['lr']}"
    return cfg

class BaseAutoencoder(L.LightningModule):
    """
    Base class for autoencoder models.

    Implements core autoencoder functionality including:
    - Weight initialization
    - Input preprocessing and normalization
    - Training/validation/test loops
    - Optimizer and learning rate scheduler configuration
    - Dead feature tracking
    - Weight normalization

    Args:
        cfg (dict): Configuration dictionary containing model hyperparameters
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        torch.manual_seed(self.cfg["seed"])

        self.b_dec = nn.Parameter(torch.zeros(self.cfg["act_size"]))
        self.b_enc = nn.Parameter(torch.zeros(self.cfg["dict_size"]))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg["act_size"], self.cfg["dict_size"])
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg["dict_size"], self.cfg["act_size"])
            )
        )
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.register_buffer('num_batches_not_active', torch.zeros((self.cfg["dict_size"],)))

        self.to(cfg["dtype"])

    def preprocess_input(self, x):
        """
        Preprocess input data by optionally normalizing.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            tuple: (normalized_x, x_mean, x_std) if normalizing, else (x, None, None)
        """
        if self.cfg.get("input_unit_norm", False):
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            if self.cfg.get("standardize_input",False):
                x_std = x.std(dim=-1, keepdim=True)
                x = x / (x_std + 1e-5)
            else:
                x_std = x.norm(dim=-1, keepdim=True)
                x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        else:
            return x, None, None

    def postprocess_output(self, x_reconstruct, x_mean, x_std):
        """
        Postprocess reconstructed output by denormalizing if needed.

        Args:
            x_reconstruct (torch.Tensor): Reconstructed output tensor
            x_mean (torch.Tensor): Mean used for normalization
            x_std (torch.Tensor): Standard deviation used for normalization

        Returns:
            torch.Tensor: Denormalized output if normalization was used, else unchanged output
        """
        if self.cfg.get("input_unit_norm", False):
            x_reconstruct = x_reconstruct * x_std + x_mean
        return x_reconstruct

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        """
        Normalize decoder weights and their gradients to have unit norm.
        """
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    def update_inactive_features(self, acts):
        """
        Update tracking of inactive features.

        Args:
            acts (torch.Tensor): Feature activations tensor
        """
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0

    def on_train_epoch_start(self):
        """Set model to training mode at start of training epoch."""
        self.train()

    def on_validation_epoch_start(self):
        """Set model to evaluation mode at start of validation epoch."""
        self.eval()

    def on_test_epoch_start(self):
        """Set model to evaluation mode at start of test epoch."""
        self.eval()

    def on_predict_epoch_start(self):
        """Set model to evaluation mode at start of prediction epoch."""
        self.eval()
    
    def training_step(self, batch, batch_idx):
        """
        Perform single training step.

        Args:
            batch: Input batch
            batch_idx (int): Batch index

        Returns:
            torch.Tensor: Loss value
        """
        x = batch
        output = self(x)
        loss = output["loss"]
        output = {
            "train_num_dead_features": output["num_dead_features"],
            "train_loss": output["loss"],
            "train_l1_loss": output["l1_loss"],
            "train_l2_loss": output["l2_loss"],
            "train_l0_norm": output["l0_norm"],
            "train_l1_norm": output["l1_norm"],
            "train_aux_loss": output["aux_loss"],
        }
        self.log_dict(output, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform single validation step.

        Args:
            batch: Input batch
            batch_idx (int): Batch index
        """
        x = batch
        output = self(x)
        output = {
            "val_num_dead_features": output["num_dead_features"],
            "val_loss": output["loss"],
            "val_l1_loss": output["l1_loss"],
            "val_l2_loss": output["l2_loss"],
            "val_l0_norm": output["l0_norm"],
            "val_l1_norm": output["l1_norm"],
            "val_aux_loss": output["aux_loss"],
        }
        self.log_dict(output, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """
        Perform single test step.

        Args:
            batch: Input batch
            batch_idx (int): Batch index
        """
        x = batch
        output = self(x)
        output = {
            "test_num_dead_features": output["num_dead_features"],
            "test_loss": output["loss"],
            "test_l1_loss": output["l1_loss"],
            "test_l2_loss": output["l2_loss"],
            "test_l0_norm": output["l0_norm"],
            "test_l1_norm": output["l1_norm"],
            "test_aux_loss": output["aux_loss"],
        }
        self.log_dict(output, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        """
        Perform single prediction step.

        Args:
            batch: Input batch
            batch_idx (int): Batch index

        Returns:
            tuple: (reconstructed_output, feature_activations)
        """
        x = batch
        output = self(x)
        return output['sae_out'], output['feature_acts']
    
    def on_before_optimizer_step(self, optimizer):
        """
        Perform operations before optimizer step.

        Args:
            optimizer: The optimizer
        """
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg["max_grad_norm"])
        self.make_decoder_weights_and_grad_unit_norm()
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate schedulers.

        Returns:
            tuple: ([optimizer], [scheduler_configs])
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg["lr"], betas=(self.cfg["beta1"], self.cfg["beta2"]), weight_decay=self.cfg["weight_decay"])

        schedulers = []
        if self.cfg['warmstart_batches']>0:
            schedulers.append({
                'scheduler': torch.optim.lr_scheduler.LinearLR(
                            optimizer,
                            start_factor=self.cfg['warmstart_start_factor'],
                            end_factor=self.cfg['warmstart_end_factor'],
                            total_iters=self.cfg['warmstart_batches'],
                            ),
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "step",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": 'WarmstartLinear',
            })

        if self.cfg['scheduler']=='RedOnPlateau':
            schedulers.append({
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                                        factor=self.cfg['reduceLROnPlateau_factor'], patience=self.cfg['reduceLROnPlateau_patience'], 
                                                                        threshold=self.cfg['reduceLROnPlateau_threshold'], threshold_mode='rel', 
                                                                        cooldown=self.cfg['reduceLROnPlateau_cooldown'], min_lr=self.cfg['reduceLROnPlateau_min'], eps=self.cfg['reduceLROnPlateau_eps']),
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "epoch",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                "monitor": "val_loss",
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a warning
                "strict": True,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": 'RedOnPlateau',
            })
        elif self.cfg['scheduler']=='OneCycleLR':
            schedulers.append({
                'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer, self.cfg['lr'], 
                                                                 epochs=self.cfg['epochs'], steps_per_epoch=self.cfg['training_set_batches']),
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "step",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": 'OneCycleLR',
            })

        elif self.cfg['scheduler']=='CosineAnnealingLR':
            schedulers.append({
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg['epochs']*self.cfg['training_set_batches']),
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "step",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": 'CosineAnnealingLR',
            })
        
        return([optimizer],schedulers)
    

class GlobalBatchTopKMatryoshkaSAE(BaseAutoencoder):
    """
    Global Batch Top-K Matryoshka Sparse Autoencoder.

    This class implements a hierarchical sparse autoencoder with multiple groups of features
    that are activated in a nested (matryoshka) fashion. Features are selected using a global
    batch-wise top-k activation mechanism.

    The model consists of multiple groups of features defined by group_sizes. Each group adds
    another layer of reconstruction on top of previous groups, allowing for hierarchical 
    feature learning.

    Args:
        cfg (dict): Configuration dictionary containing:
            - group_sizes (list): List of integers defining size of each feature group
            - act_size (int): Size of input/output activation dimension
            - device (str): Device to place model on ('cpu' or 'cuda')
            - dtype (torch.dtype): Data type for model parameters
            - top_k (int): Number of top activations to keep per batch
            - top_k_aux (int): Number of auxiliary activations for dead feature recovery
            - n_batches_to_dead (int): Number of batches before feature considered dead
            - l1_coeff (float): L1 regularization coefficient
            - aux_penalty (float): Penalty coefficient for auxiliary loss
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        total_dict_size = sum(cfg["group_sizes"])
        self.group_sizes = cfg["group_sizes"]
        
        self.group_indices = [0] + list(torch.cumsum(torch.tensor(cfg["group_sizes"]), dim=0))
        self.active_groups = len(cfg["group_sizes"])

        self.b_dec = nn.Parameter(torch.zeros(self.config["act_size"]))
        self.b_enc = nn.Parameter(torch.zeros(total_dict_size))
        
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(cfg["act_size"], total_dict_size)
            )
        )
        
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(total_dict_size, cfg["act_size"])
            )
        )
        
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.num_batches_not_active = torch.zeros(total_dict_size, device=cfg["device"])
        self.register_buffer('threshold', torch.tensor(0.0))
        self.to(cfg["dtype"]).to(cfg["device"])

    def compute_activations(self, x_cent):
        """
        Compute activations using global batch top-k sparsity.

        Args:
            x_cent (torch.Tensor): Centered input tensor (input - bias)

        Returns:
            tuple: (acts, acts_topk)
                - acts: Raw ReLU activations
                - acts_topk: Sparse activations after top-k selection
        """
        pre_acts = x_cent @ self.W_enc
        acts = F.relu(pre_acts)
        
        if self.training:
            acts_topk = torch.topk(
                acts.flatten(), 
                self.cfg["top_k"] * x_cent.shape[0], 
                dim=-1
            )
            acts_topk = (
                torch.zeros_like(acts.flatten())
                .scatter(-1, acts_topk.indices, acts_topk.values)
                .reshape(acts.shape)
            )
            self.update_threshold(acts_topk)
        else:
            acts_topk = torch.where(acts > self.threshold, acts, torch.zeros_like(acts))
        
        return acts, acts_topk


    def forward(self, x):
        """
        Forward pass through the network.

        Processes input through each group sequentially, building up the reconstruction
        layer by layer.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            dict: Output dictionary containing reconstructions and loss metrics
        """
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        x_reconstruct = self.b_dec

        intermediate_reconstructs = []
        all_acts, all_acts_topk = self.compute_activations(x_cent)

        for i in range(self.active_groups):
            start_idx = self.group_indices[i]
            end_idx = self.group_indices[i+1]
            W_dec_slice = self.W_dec[start_idx:end_idx, :]
            acts_topk = all_acts_topk[:, start_idx:end_idx]
            x_reconstruct = acts_topk @ W_dec_slice + x_reconstruct
            intermediate_reconstructs.append(x_reconstruct)

        self.update_inactive_features(all_acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, all_acts, all_acts_topk, x_mean, 
                                  x_std, intermediate_reconstructs)
        return output

    def get_loss_dict(self, x, x_reconstruct, all_acts, all_acts_topk, x_mean, x_std, intermediate_reconstructs):
        """
        Compute all loss terms and metrics.

        Args:
            x (torch.Tensor): Input tensor
            x_reconstruct (torch.Tensor): Final reconstruction
            all_acts (torch.Tensor): Raw activations
            all_acts_topk (torch.Tensor): Sparse activations
            x_mean (torch.Tensor): Input mean for normalization
            x_std (torch.Tensor): Input std for normalization
            intermediate_reconstructs (list): List of intermediate reconstructions

        Returns:
            dict: Dictionary containing all loss terms and metrics
        """
        total_l2_loss = (self.b_dec - x.float()).pow(2).mean()
        l2_losses = torch.tensor([]).to(x.device)
        for intermediate_reconstruct in intermediate_reconstructs:
            l2_losses = torch.cat([l2_losses, (intermediate_reconstruct.float() - 
                                             x.float()).pow(2).mean().unsqueeze(0)])
            total_l2_loss += (intermediate_reconstruct.float() - x.float()).pow(2).mean()

        min_l2_loss = l2_losses.min()
        max_l2_loss = l2_losses.max()
        mean_l2_loss = total_l2_loss / (len(intermediate_reconstructs) + 1)

        l1_norm = all_acts_topk.float().abs().sum(-1).mean()
        l0_norm = (all_acts_topk > 0).float().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, all_acts)
        loss = mean_l2_loss + l1_loss + aux_loss
        
        num_dead_features = (self.num_batches_not_active > self.cfg["n_batches_to_dead"]).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": all_acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": mean_l2_loss,
            "min_l2_loss": min_l2_loss,
            "max_l2_loss": max_l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
            "threshold": self.threshold,
        }
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, all_acts):
        """
        Compute auxiliary loss for dead feature recovery.

        Args:
            x (torch.Tensor): Input tensor
            x_reconstruct (torch.Tensor): Reconstruction tensor
            all_acts (torch.Tensor): All activations before sparsification

        Returns:
            torch.Tensor: Auxiliary loss value
        """
        residual = x.float() - x_reconstruct.float()
        aux_reconstruct = torch.zeros_like(residual)
        
        acts = all_acts
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        
        if dead_features.sum() > 0:
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            aux_reconstruct = aux_reconstruct + x_reconstruct_aux
                
        if aux_reconstruct.abs().sum() > 0:
            aux_loss = self.cfg["aux_penalty"] * (aux_reconstruct.float() - residual.float()).pow(2).mean()
            return aux_loss
        else:
            return torch.tensor(0.0, device=x.device)
    
    @torch.no_grad()
    def update_threshold(self, acts_topk, lr=0.01):
        """
        Update activation threshold using exponential moving average.

        Args:
            acts_topk (torch.Tensor): Top-k activations
            lr (float, optional): Learning rate for update. Defaults to 0.01.
        """
        positive_mask = acts_topk > 0
        if positive_mask.any():
            min_positive = acts_topk[positive_mask].min()
            self.threshold = (1 - lr) * self.threshold + lr * min_positive


class BatchTopKSAE(BaseAutoencoder):
    """
    Batch-wise top-k sparse autoencoder.

    This class implements a sparse autoencoder that selects the top-k activations across the entire batch
    rather than per individual sample. This encourages competition between features across samples and can
    lead to more efficient feature learning.

    The model consists of:
    - An encoder that maps inputs to a higher dimensional feature space
    - A sparsification step that keeps only the top-k activations across the batch
    - A decoder that reconstructs the input from the sparse features
    - An auxiliary loss mechanism to reactivate dead features
    - A dynamic threshold that adapts to activation magnitudes during training

    Args:
        cfg (dict): Configuration dictionary containing:
            - act_size (int): Size of input/output activation dimension
            - dict_size (int): Number of features in dictionary
            - top_k (int): Number of top activations to keep per batch
            - top_k_aux (int): Number of auxiliary activations for dead feature recovery
            - n_batches_to_dead (int): Number of batches before feature considered dead
            - l1_coeff (float): L1 regularization coefficient
            - aux_penalty (float): Penalty coefficient for auxiliary loss
            - device (str): Device to place model on ('cpu' or 'cuda')
            - dtype (torch.dtype): Data type for model parameters
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.register_buffer('threshold', torch.tensor(0.0))
        
    def compute_activations(self, x):
        """
        Compute activations using batch-wise top-k sparsity.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            tuple: (acts, acts_topk)
                - acts: Raw ReLU activations
                - acts_topk: Sparse activations after top-k selection
        """
        x_cent = x - self.b_dec
        pre_acts = x_cent @ self.W_enc
        acts = F.relu(pre_acts)
        
        if self.training:
            acts_topk = torch.topk(
                acts.flatten(), 
                self.config["top_k"] * x.shape[0], 
                dim=-1
            )
            acts_topk = (
                torch.zeros_like(acts.flatten())
                .scatter(-1, acts_topk.indices, acts_topk.values)
                .reshape(acts.shape)
            )
        else:
            acts_topk = torch.where(acts > self.threshold, acts, torch.zeros_like(acts))
        
        return acts, acts_topk

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Performs the following steps:
        1. Preprocesses input data (centering/normalization)
        2. Computes activations through encoder
        3. Applies batch-wise top-k sparsification
        4. Reconstructs input through decoder
        5. Updates threshold and tracks inactive features
        6. Computes loss terms and metrics

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, act_size)

        Returns:
            dict: Output dictionary containing:
                - sae_out (torch.Tensor): Reconstructed output
                - feature_acts (torch.Tensor): Sparse feature activations
                - num_dead_features (int): Number of inactive features
                - loss (torch.Tensor): Total loss
                - l1_loss (torch.Tensor): L1 regularization loss
                - l2_loss (torch.Tensor): Reconstruction loss
                - l0_norm (torch.Tensor): Number of non-zero activations
                - l1_norm (torch.Tensor): Sum of absolute activations
                - aux_loss (torch.Tensor): Auxiliary loss for dead features
                - threshold (torch.Tensor): Current activation threshold
        """
        x, x_mean, x_std = self.preprocess_input(x)
        acts, acts_topk = self.compute_activations(x)
        
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec

        self.update_threshold(acts_topk)
        self.update_inactive_features(acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        """
        Calculate loss terms and metrics.

        Computes various loss components and metrics including:
        - L2 reconstruction loss
        - L1 sparsity regularization
        - L0 sparsity measure (number of non-zero activations)
        - Auxiliary loss for dead feature reactivation

        Args:
            x (torch.Tensor): Input tensor
            x_reconstruct (torch.Tensor): Reconstructed output
            acts (torch.Tensor): Pre-sparsity activations
            acts_topk (torch.Tensor): Sparse activations after top-k
            x_mean (torch.Tensor): Input mean for denormalization
            x_std (torch.Tensor): Input std for denormalization

        Returns:
            dict: Dictionary containing loss terms and metrics
        """
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        loss = l2_loss + l1_loss + aux_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
            "threshold": self.threshold,
        }
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        """
        Calculate auxiliary loss for dead feature reactivation.

        This loss encourages dead features to learn to reconstruct the residual error
        of the main reconstruction. A feature is considered dead if it hasn't been
        activated for n_batches_to_dead batches.

        Args:
            x (torch.Tensor): Input tensor
            x_reconstruct (torch.Tensor): Reconstructed output from main features
            acts (torch.Tensor): All feature activations before sparsification

        Returns:
            torch.Tensor: Auxiliary loss value for dead feature reactivation
        """
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)

class TopKSAE(BaseAutoencoder):
    """
    Sample-wise top-k sparse autoencoder.

    Takes top k activations per sample.

    Args:
        cfg (dict): Configuration dictionary
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            dict: Output dictionary containing reconstructed data and metrics
        """
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        acts_topk = torch.topk(acts, self.cfg["top_k"], dim=-1)
        acts_topk = torch.zeros_like(acts).scatter(
            -1, acts_topk.indices, acts_topk.values
        )
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec

        self.update_inactive_features(acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        """
        Calculate loss terms.

        Args:
            x (torch.Tensor): Input tensor
            x_reconstruct (torch.Tensor): Reconstructed output
            acts (torch.Tensor): Pre-sparsity activations
            acts_topk (torch.Tensor): Sparse activations after top-k
            x_mean (torch.Tensor): Input mean for denormalization
            x_std (torch.Tensor): Input std for denormalization

        Returns:
            dict: Dictionary of loss terms and metrics
        """
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        loss = l2_loss + l1_loss + aux_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
        }
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        """
        Calculate auxiliary loss for dead feature reactivation.

        Args:
            x (torch.Tensor): Input tensor
            x_reconstruct (torch.Tensor): Reconstructed output
            acts (torch.Tensor): Feature activations

        Returns:
            torch.Tensor: Auxiliary loss value
        """
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)

class VanillaSAE(BaseAutoencoder):
    """
    Basic sparse autoencoder without top-k sparsification.

    Uses L1 regularization for sparsity.

    Args:
        cfg (dict): Configuration dictionary
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            dict: Output dictionary containing reconstructed data and metrics
        """
        x, x_mean, x_std = self.preprocess_input(x)
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        self.update_inactive_features(acts)
        output = self.get_loss_dict(x, x_reconstruct, acts, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, x_mean, x_std):
        """
        Calculate loss terms.

        Args:
            x (torch.Tensor): Input tensor
            x_reconstruct (torch.Tensor): Reconstructed output
            acts (torch.Tensor): Feature activations
            x_mean (torch.Tensor): Input mean for denormalization
            x_std (torch.Tensor): Input std for denormalization

        Returns:
            dict: Dictionary of loss terms and metrics
        """
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts > 0).float().sum(-1).mean()
        loss = l2_loss + l1_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()

        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": 0,
        }
        return output

import torch
import torch.nn as nn
import torch.autograd as autograd

class RectangleFunction(autograd.Function):
    """
    Custom autograd function implementing a rectangular window function.
    """

    @staticmethod
    def forward(ctx, x):
        """
        Forward pass returns 1 for inputs between -0.5 and 0.5, 0 otherwise.

        Args:
            ctx: Context for saving tensors
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Binary output tensor
        """
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass zeroes gradients outside window.

        Args:
            ctx: Context containing saved tensors
            grad_output (torch.Tensor): Gradient from downstream

        Returns:
            torch.Tensor: Modified gradient
        """
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input

class JumpReLUFunction(autograd.Function):
    """
    Custom autograd function implementing JumpReLU activation.
    """

    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        """
        Forward pass implementing JumpReLU.

        Args:
            ctx: Context for saving tensors
            x (torch.Tensor): Input tensor
            log_threshold (torch.Tensor): Log of threshold parameter
            bandwidth (float): Bandwidth parameter

        Returns:
            torch.Tensor: Output tensor
        """
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for JumpReLU.

        Args:
            ctx: Context containing saved tensors
            grad_output (torch.Tensor): Gradient from downstream

        Returns:
            tuple: (dx, dlog_threshold, None)
        """
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth

class JumpReLU(nn.Module):
    """
    JumpReLU module implementing learnable threshold activation.

    Args:
        feature_size (int): Number of features
        bandwidth (float): Bandwidth parameter
    """

    def __init__(self, feature_size, bandwidth):
        super(JumpReLU, self).__init__()
        self.log_threshold = nn.Parameter(torch.zeros(feature_size))
        self.bandwidth = bandwidth

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output after JumpReLU activation
        """
        return JumpReLUFunction.apply(x, self.log_threshold, self.bandwidth)

class StepFunction(autograd.Function):
    """
    Custom autograd function implementing a step function.
    """

    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        """
        Forward pass implementing step function.

        Args:
            ctx: Context for saving tensors
            x (torch.Tensor): Input tensor
            log_threshold (torch.Tensor): Log of threshold parameter
            bandwidth (float): Bandwidth parameter

        Returns:
            torch.Tensor: Binary output tensor
        """
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for step function.

        Args:
            ctx: Context containing saved tensors
            grad_output (torch.Tensor): Gradient from downstream

        Returns:
            tuple: (dx, dlog_threshold, None)
        """
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth

class JumpReLUSAE(BaseAutoencoder):
    """
    Sparse autoencoder using JumpReLU activation.

    Args:
        cfg (dict): Configuration dictionary
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.jumprelu = JumpReLU(feature_size=cfg["dict_size"], bandwidth=cfg["bandwidth"])

    def forward(self, x, use_pre_enc_bias=False):
        x, x_mean, x_std = self.preprocess_input(x)

        if use_pre_enc_bias:
            x = x - self.b_dec

        pre_activations = torch.relu(x @ self.W_enc + self.b_enc)
        feature_magnitudes = self.jumprelu(pre_activations)

        x_reconstructed = feature_magnitudes @ self.W_dec + self.b_dec

        return self.get_loss_dict(x, x_reconstructed, feature_magnitudes, x_mean, x_std)

    def get_loss_dict(self, x, x_reconstruct, acts, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()

        l0 = StepFunction.apply(acts, self.jumprelu.log_threshold, self.cfg["bandwidth"]).sum(dim=-1).mean()
        l0_loss = self.cfg["l1_coeff"] * l0
        l1_loss = l0_loss

        loss = l2_loss + l1_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()

        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0,
            "l1_norm": l0,
            "aux_loss": 0,
        }
        return output