import lightning.pytorch as L


class SAETraining:
    """Training wrapper class for Sparse Autoencoders using PyTorch Lightning.

    This class handles training setup and execution including callbacks, logging, and training loops.

    Args:
        cfg (dict): Configuration dictionary containing training parameters
        custom_callbacks (list, optional): List of additional PyTorch Lightning callbacks. Defaults to [].

    Attributes:
        cfg (dict): Configuration dictionary
        custom_callbacks (list): List of custom callbacks
        trainer (lightning.Trainer): PyTorch Lightning trainer instance
    """

    def __init__(self, cfg, custom_callbacks=[]):
        self.cfg = cfg
        self.custom_callbacks = custom_callbacks

        L.seed_everything(self.cfg['seed'], workers=True) # sets seeds for numpy, torch and python.random.
        callbacks = []
        
        if self.cfg['include_checkpointing']:
            checkpoint_callback = L.callbacks.ModelCheckpoint(dirpath=f"{self.cfg['outpath']}", 
                                                                filename=f"{self.cfg['name']}_{self.cfg['seed']}"+"_{epoch:02d}_{val_loss:.2f}", 
                                                                save_top_k=1, monitor="val_loss", mode="min")
            callbacks = callbacks+[checkpoint_callback]

        if self.cfg['include_early_stopping']:
            early_stop_callback = L.callbacks.EarlyStopping(monitor="val_loss", min_delta=self.cfg['min_delta'], patience=self.cfg['patience'], verbose=False, mode="min")
            callbacks = callbacks+[early_stop_callback]

        if self.cfg['track_LR']:
            LR_monitor_callback = L.callbacks.LearningRateMonitor()
            callbacks.append(LR_monitor_callback)
        
        callbacks = callbacks+self.custom_callbacks

        wandb_logger = None
        if not self.cfg['name'] is None:
            arg_dict = self.cfg
            try:
                import wandb
                wandb.login()
                wandb_logger = L.loggers.WandbLogger(project=self.cfg['wandb_project'],
                                                    group = self.cfg['name'],
                                                    config=arg_dict)
            except wandb.errors.CommError:
                print('Hit wandb init error! Retrying without logging')
                wandb_logger=None

        self.trainer = L.Trainer(deterministic=True, 
                        logger=wandb_logger,
                        accelerator = self.cfg['accelerator'], devices= self.cfg['devices'],
                        max_epochs=self.cfg['epochs'], 
                        callbacks=callbacks,)
        

    def train(self, model, train_dl, val_dl):
        """Train the model.

        Args:
            model (lightning.LightningModule): Model to train
            train_dl (DataLoader): Training data loader
            val_dl (DataLoader): Validation data loader

        Returns:
            lightning.LightningModule: Trained model
        """
        self.trainer.validate(model=model, dataloaders=val_dl)
        self.trainer.fit(model, train_dl, val_dl)
        return(model)
    

    def validate(self, val_dl, model=None):
        """Run validation on the model.

        Args:
            val_dl (DataLoader): Validation data loader
            model (lightning.LightningModule, optional): Model to validate. If None, loads from best checkpoint. Defaults to None.

        Returns:
            list: Validation metrics
        """
        if model is not None:
            val_metrics = self.trainer.validate(model=model, dataloaders=val_dl)
        else:
            val_metrics = self.trainer.validate(dataloaders = val_dl, ckpt_path='best')
        return(val_metrics)

    def test(self, test_dataloader, model=None):
        """Run testing on the model.

        Args:
            test_dataloader (DataLoader): Test data loader
            model (lightning.LightningModule, optional): Model to test. If None, loads from best checkpoint. Defaults to None.

        Returns:
            list: Test metrics
        """
        if model is not None:
            test_metrics = self.trainer.test(dataloaders = test_dataloader, model=model)
        else:
            test_metrics = self.trainer.test(dataloaders = test_dataloader, ckpt_path='best')
        return(test_metrics)
