# Import required libraries
import pandas as pd
from Bio import SeqIO
from itertools import cycle, islice
import numpy as np
import traceback
import lightning.pytorch as L
from torch.utils.data import DataLoader
import torch

class LiaoDataset():
    """A dataset class for handling DNA sequence data with exons and barcodes.
    
    This class processes sequences from a plasmid template and returns sequence-PSI pairs.
    It handles loading sequence data from CSV files and plasmid templates, finding exon and
    barcode positions, and generating batches of processed sequences.

    Attributes:
        data (pd.DataFrame): DataFrame containing sequence data
        batch_size (int): Size of batches to return
        transform_x (callable, optional): Transform to apply to input sequences
        preload (bool): Whether to preload all sequences into memory
        preloaded_data (list): Cached sequence data if preloaded
        plasmid (str): Plasmid template sequence
        ex_pos (int): Position of exon in plasmid
        ex_len (int): Length of exon sequence
        bc_pos (int): Position of barcode in plasmid  
        bc_len (int): Length of barcode sequence
        context_len (int): Length of context sequence to include
    """

    def __init__(self, csv_path, plasmid_path, context_len, auto_find_bc_pos=True,
                 auto_find_ex_pos = True,
                 batch_size=64, transform_x=None, preload=False):
        """Initialize the dataset.

        Args:
            csv_path (str): Path to CSV file containing sequence data
            plasmid_path (str): Path to plasmid template file in GenBank format
            context_len (int): Length of context sequence to include
            auto_find_bc_pos (bool, optional): Whether to automatically find barcode position. Defaults to True.
            auto_find_ex_pos (bool, optional): Whether to automatically find exon position. Defaults to True.
            batch_size (int, optional): Size of batches to return. Defaults to 64.
            transform_x (callable, optional): Transform to apply to input sequences. Defaults to None.
            preload (bool, optional): Whether to preload all sequences into memory. Defaults to False.
        """
        self.data = pd.read_csv(csv_path)
        self.batch_size = batch_size
        self.transform_x = transform_x
        self.preload = preload
        self.preloaded_data = None
        
        # Read plasmid template sequence
        with open(plasmid_path, "r") as f:
            gbk = SeqIO.parse(f, 'genbank')
            self.plasmid = str(next(gbk).seq).upper()
        
        # Automatically find exon position if requested
        if auto_find_ex_pos:
            ex_len = len(self.data.iloc[0]['exon'])

            # Find position of N's marking exon location
            ex_pos = self.plasmid.find(''.join(['N']*ex_len))

            self.set_exon_pos(ex_pos, ex_len)


        # Automatically find barcode position if requested
        if auto_find_bc_pos:
            bc_len = len(self.data.iloc[0]['barcode'])

            # Find position of N's marking exon location
            bc_pos = self.plasmid.find(''.join(['N']*bc_len))

            self.set_bc_pos(bc_pos, bc_len)

        self.context_len = context_len

    def set_exon_pos(self, ex_start, ex_len):
        """Set exon position and length manually.
        
        Args:
            ex_start (int): Starting position of exon in plasmid
            ex_len (int): Length of exon sequence

        Returns:
            LiaoDataset: Returns self for method chaining

        Raises:
            AssertionError: If exon length is 0 or position is negative
        """
        self.ex_pos = ex_start
        self.ex_len = ex_len

        assert self.ex_len>0, 'Exon length is 0'
        assert self.ex_pos>=0, 'exon pos is negative'
        return(self)
    
    def set_bc_pos(self, bc_start, bc_len):
        """Set barcode position and length manually.

        Args:
            bc_start (int): Starting position of barcode in plasmid
            bc_len (int): Length of barcode sequence

        Returns:
            LiaoDataset: Returns self for method chaining

        Raises:
            AssertionError: If barcode length is 0 or position is negative
        """
        self.bc_pos = bc_start
        self.bc_len = bc_len

        assert self.bc_len>0, 'Barcode length is 0'
        assert self.bc_pos>=0, 'barcode pos is negative'
        return(self)

    def __len__(self):
        """Return number of sequences in dataset.

        Returns:
            int: Number of sequences in the dataset
        """
        return(len(self.data))
    
    def process_x(self, row):
        """Process a sequence by replacing placeholders in plasmid template.
        
        Args:
            row (pd.Series): Row containing exon and barcode sequences
            
        Returns:
            numpy.ndarray: Processed sequence with context
        """
        
        var = self.plasmid

        # Replace N's with actual exon sequence if it will be included in the context
        if len(var)-self.ex_len < self.context_len:
            var = var[:self.ex_pos]+row['exon']+var[self.ex_pos+self.ex_len:]

        # Replace N's with actual barcode sequence if it will be included in the context
        if self.ex_pos+self.ex_len+self.context_len > self.bc_pos or self.context_len-self.ex_pos > len(var)-self.bc_pos+self.bc_len:
            var = var[:self.bc_pos]+row['barcode']+var[self.bc_pos+self.bc_len:]

        # Get right context by cycling sequence
        right = list(islice(cycle(var), self.ex_pos+self.ex_len, self.context_len//2+self.ex_pos+self.ex_len))
        flip_pos = len(self.plasmid)-self.ex_pos
        # Get left context by cycling reversed sequence
        left = list(islice(cycle(var[::-1]), flip_pos, self.context_len//2+flip_pos))[::-1]

        # Combine left context, exon region, and right context
        var = left + list(row['exon']) + right

        var_ar = np.array(var)
        return(var_ar)
    
    def __getitem__(self, index):
        """Get sequence-label pairs.

        Args:
            index (int or slice or list or numpy.ndarray): Index, slice, or array of indices to retrieve samples.
                For int: returns single sample
                For slice: returns samples in range
                For list/array: returns samples at specified indices

        Returns:
            tuple: For single index: (sequence, label) pair where
                sequence (numpy.ndarray): Processed DNA sequence with context
                label (numpy.ndarray): Target value (PSI or logit transformed PSI)
                For multiple indices: Batched sequences and labels via collate_fn

        Raises:
            IndexError: If index is out of range
            ValueError: If index type is invalid
        """
        # Handle slice and fancy indexing
        if isinstance(index, slice):
            start = index.start if index.start is not None else 0
            stop = index.stop if index.stop is not None else len(self)
            step = index.step if index.step is not None else 1
            indices = range(start, stop, step)
            return self.collate_fn([self[i] for i in indices])
        elif isinstance(index, (list, np.ndarray)):
            return self.collate_fn([self[i] for i in index])
        elif isinstance(index, int) or isinstance(index, np.integer):
            if index >= len(self):
                raise IndexError
            if self.preloaded_data is not None:
                return self.preloaded_data[index]
            
            row = self.data.loc[index]
            x = self.process_x(row)
            if self.transform_x is not None:
                x = self.transform_x(x)
        
            
            return(x)
        
        else:
            raise ValueError(f"Invalid index type: {type(index)}")
    
    def collate_fn(self, batch):
        """Collate function for batching sequences and labels.
        
        Args:
            batch (list): List of sequence-label pairs

        Returns:
            tuple: Batch of processed sequences and labels as numpy arrays
        """
        return(tuple(np.stack(n) for n in zip(*batch)))
    
    def batches(self):
        """Generate batches of sequence-label pairs.

        Yields:
            tuple: Batch of sequence-label pairs
        """
        it = iter(self)
        while batch := tuple(islice(it, self.batch_size)):
            yield self.collate_fn(batch)

    def __enter__(self):
        """Context manager entry.

        Returns:
            LiaoDataset: Returns self
        """
        self.open()
        return(self)
    
    def __exit__(self, exc_type, exc_value, tb):
        """Context manager exit with exception handling.

        Args:
            exc_type: Exception type if error occurred
            exc_value: Exception value if error occurred  
            tb: Traceback if error occurred

        Returns:
            bool: False if exception occurred, True otherwise
        """
        self.close()
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            return(False)
        return(True)

    def preload_data(self):
        """Preload data into memory.

        Returns:
            list: List of processed sequence-label pairs
        """
        return([self.__getitem__(i) for i in range(len(self))])
    
    def open(self):
        """Initialize resources and preload data if requested."""
        if self.preload:
            self.preloaded_data = self.preload_data()
    
    def close(self):
        """Cleanup resources and clear preloaded data."""
        if self.preloaded_data is not None:
            self.preloaded_data = None

class LiaoDatasetEmbedded(LiaoDataset):
    """A dataset class for handling DNA sequence data with exons and barcodes.
    
    This class extends LiaoDataset to process sequences through an embedding model
    before returning them. It can preload embeddings for faster access.

    Attributes:
        ds (LiaoDataset): Base dataset instance
        preload_embeddings (bool): Whether to preload all embeddings
        trainer (lightning.Trainer): PyTorch Lightning trainer
        model: Model to generate embeddings
        embeddings (torch.Tensor): Cached embeddings if preloaded
        num_workers (int): Number of workers for data loading
        preload (bool): Whether to preload data
        data (pd.DataFrame): DataFrame containing sequence data
        preloaded_data: Cached sequence data if preloaded
    """

    def __init__(self, csv_path, plasmid_path, context_len, model, auto_find_bc_pos=True,
                 auto_find_ex_pos = True,
                 batch_size=64, transform_x=None, preload=False, preload_embeddings=False, 
                 trainer=None, num_workers=16):
        """Initialize the dataset.

        Args:
            csv_path (str): Path to CSV file containing sequence data
            plasmid_path (str): Path to plasmid template file
            context_len (int): Length of context sequence
            model: Model to generate embeddings
            auto_find_bc_pos (bool, optional): Whether to find barcode position. Defaults to True.
            auto_find_ex_pos (bool, optional): Whether to find exon position. Defaults to True.
            batch_size (int, optional): Batch size. Defaults to 64.
            transform_x (callable, optional): Transform for sequences. Defaults to None.
            preload (bool, optional): Whether to preload sequences. Defaults to False.
            preload_embeddings (bool, optional): Whether to preload embeddings. Defaults to False.
            trainer (lightning.Trainer, optional): PyTorch Lightning trainer. Defaults to None.
            num_workers (int, optional): Number of workers for data loading. Defaults to 16.
        """
        self.ds = LiaoDataset(csv_path, plasmid_path, context_len, auto_find_bc_pos=auto_find_bc_pos,
                         auto_find_ex_pos=auto_find_ex_pos, batch_size=batch_size, transform_x=transform_x, preload=preload)
        
        self.preload_embeddings = preload_embeddings
        if trainer is None:
            trainer = L.Trainer()
        self.trainer = trainer
        self.model = model
        self.embeddings = None
        self.num_workers = num_workers
        self.preload = preload
        self.data = self.ds.data
        self.preloaded_data = self.ds.preloaded_data

    def process_x(self, row, idx):
        """Process a sequence by generating its embedding.
        
        Args:
            row (pd.Series): Row containing sequence data
            idx (int): Index of the sequence
            
        Returns:
            torch.Tensor: Embedding of the sequence
        """
        if self.embeddings is None and not self.preload_embeddings:
            x = self.ds.process_x(row)
            embedding = self.model.embed(x)
        elif self.embeddings is None:
            dl = DataLoader(self.ds, batch_size=self.ds.batch_size, num_workers=self.num_workers)
            self.embeddings = torch.cat(self.trainer.predict(self.model, dl), dim=0)
            embedding = self.embeddings[idx]
        else:
            embedding = self.embeddings[idx]
        return(embedding)
    
    def __getitem__(self, index):
        """Get embedded sequence-label pairs.

        Args:
            index (int or slice or list or numpy.ndarray): Index, slice, or array of indices

        Returns:
            tuple: For single index: (embedding, label) pair where
                embedding (torch.Tensor): Sequence embedding
                label (numpy.ndarray): Target value
                For multiple indices: Batched embeddings and labels

        Raises:
            IndexError: If index is out of range
            ValueError: If index type is invalid
        """
        # Handle slice and fancy indexing
        if isinstance(index, slice):
            start = index.start if index.start is not None else 0
            stop = index.stop if index.stop is not None else len(self)
            step = index.step if index.step is not None else 1
            indices = range(start, stop, step)
            return self.collate_fn([self[i] for i in indices])
        elif isinstance(index, (list, np.ndarray)):
            return self.collate_fn([self[i] for i in index])
        elif isinstance(index, int) or isinstance(index, np.integer):
            if index >= len(self):
                raise IndexError
            if self.preloaded_data is not None:
                return self.preloaded_data[index]
            
            row = self.ds.data.loc[index]
            x = self.process_x(row, index)
            
            return(x)
        
        else:
            raise ValueError(f"Invalid index type: {type(index)}")
