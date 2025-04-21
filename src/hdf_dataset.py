import h5py
import numpy as np
import torch
from math import ceil


class HDF3DDataset:
    """Iterator class for accessing first two dimensions of a 3D HDF5 dataset with chunk caching."""
    
    def __init__(self, file_path, dataset_name, transform=None, chunk_size=1000):
        """Initialize the iterator with caching parameters.
        
        Args:
            file_path (str): Path to the HDF5 file
            dataset_name (str): Name of the dataset within the HDF5 file
            transform: Optional transform to apply to data
            chunk_size (int): Size of chunks to cache
            cache_margin (int): Extra margin around chunk to cache
        """
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.transform = transform
        self.chunk_size = chunk_size
        
        with h5py.File(file_path, 'r') as f:
            self.shape = f[dataset_name].shape
            if len(self.shape) != 3:
                raise ValueError("Dataset must be 3-dimensional")

        self.hdf_file = None
        self.current_idx = 0
        self.cached_data = None
        self.cache_start_idx = None
        self.cache_end_idx = None

    def __len__(self):
        """Return the total number of items in first two dimensions."""
        return self.shape[0] * self.shape[1]

    def _get_chunk_bounds(self, idx):
        """Calculate chunk boundaries for given index."""
        if idx >= len(self):
            raise IndexError("Index out of bounds")
        if idx < 0:
            raise IndexError("Index out of bounds")
        
        # Round chunk_start down to start of row
        row_start = (idx) // self.shape[1]
        chunk_start = row_start * self.shape[1]
        
        
        # Round chunk_end up to end of row
        row_end = ceil((chunk_start + self.chunk_size) / self.shape[1])
        chunk_end = min(len(self), row_end * self.shape[1])

        return chunk_start, chunk_end

    def _load_chunk(self, idx):
        """Load a new chunk of data into cache."""
        chunk_start, chunk_end = self._get_chunk_bounds(idx)
            
        # Convert flat indices to 2D coordinates
        start_i = chunk_start // self.shape[1]
        start_j = chunk_start % self.shape[1]
        end_i = chunk_end // self.shape[1]
        end_j = chunk_end % self.shape[1]

        # Verify chunk boundaries align with row boundaries
        if start_j != 0:
            raise ValueError(f"Chunk start not aligned to row boundary (start_j = {start_j})")
        if end_j != 0 and chunk_end != len(self):
            raise ValueError(f"Chunk end not aligned to row boundary (end_j = {end_j})")
        
        # Load the chunk
        self.cached_data = self.hdf_file[self.dataset_name][start_i:end_i, :, :]
        
        # Set cache boundaries
        self.cache_start_idx = chunk_start
        self.cache_end_idx = chunk_end

    def __getitem__(self, idx):
        """Get item at specified index with caching."""
        if isinstance(idx, (slice, list, np.ndarray)):
            if isinstance(idx, slice):
                start = idx.start or 0
                stop = idx.stop or len(self)
                step = idx.step or 1
                results = [self[i] for i in range(start, stop, step)]
            else:
                results = [self[i] for i in idx]
            return np.stack(results)
            
        if isinstance(idx, tuple) and len(idx) == 2:
            i, j = idx
            if i >= self.shape[0] or j >= self.shape[1]:
                raise IndexError("Index out of bounds")
            idx = i * self.shape[1] + j
        elif isinstance(idx, int):
            if idx >= len(self):
                raise IndexError("Index out of bounds")
        else:
            raise ValueError("Index must be int, list, array, or 2-tuple")

        # Check if index is not in cached chunk
        if (self.cached_data is None or 
            idx < self.cache_start_idx or 
            idx >= self.cache_end_idx):
            self._load_chunk(idx)
        
        # Convert to chunk-relative coordinates
        rel_idx = idx - self.cache_start_idx
        i = rel_idx // self.shape[1]
        j = rel_idx % self.shape[1]
        
        x = self.cached_data[i, j, :]
        
        if self.transform is not None:
            x = self.transform(x)
        return x

    def __iter__(self):
        """Initialize iterator."""
        self.current_idx = 0
        return self
        
    def __next__(self):
        """Get next item in iteration."""
        if self.current_idx >= len(self):
            raise StopIteration
            
        item = self[self.current_idx]
        self.current_idx += 1
        return item

    def close(self):
        """Close file and clear cache."""
        if self.hdf_file is not None:
            self.hdf_file.close()
            self.hdf_file = None
        self.cached_data = None
        self.cache_start_idx = None 
        self.cache_end_idx = None

    def open(self):
        """Open the HDF5 file."""
        self.close()
        self.hdf_file = h5py.File(self.file_path, 'r')

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def worker_init_fn(self, worker_id):
        """Initialize worker."""
        self._worker_id = worker_id
        self.open()

        
class ChunkBatchSampler(torch.utils.data.Sampler):
    """Sampler that returns batches of indices within the same chunk."""
    
    def __init__(self, dataset, batch_size, shuffle=True, split_ratio=None, split='train', seed=None):
        """Initialize the sampler.
        
        Args:
            dataset: HDF3DIterator dataset
            batch_size: Size of batches to return
            shuffle: Whether to shuffle indices within chunks
            split_ratio: If not None, ratio to split chunks into train/val (e.g. 0.8)
            split: Which split to use ('train' or 'val'), only used if split_ratio is set
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        # Get all chunk boundaries
        self.chunks = []
        pos = 0
        while pos < len(dataset):
            chunk_start, chunk_end = dataset._get_chunk_bounds(pos)
            self.chunks.append((chunk_start, chunk_end))
            pos = chunk_end + 1

        # check that the chunks are contiguous and not overlapping
        for i in range(len(self.chunks)-1):
            if self.chunks[i][1] != self.chunks[i+1][0]:
                raise ValueError("Chunks are not contiguous")
            if self.chunks[i][1] > self.chunks[i+1][0]:
                raise ValueError("Chunks are overlapping")
        #Check that the chunks cover the entire dataset
        if (self.chunks[-1][1] != len(self.dataset))&(self.chunks[0][0] != 0):
            raise ValueError("Chunks do not cover the entire dataset")
        
        # check that the chunk lengths are multiple of the batch size
        for chunk_start, chunk_end in self.chunks[:-1]:
            if (chunk_end - chunk_start) % self.batch_size != 0:
                raise ValueError("Chunk length is not a multiple of the batch size")
            
            
        # Split chunks into train/val if requested
        if split_ratio is not None:
            if split not in ['train', 'val']:
                raise ValueError("split must be 'train' or 'val'")
                
            # Randomly assign chunks to train/val
            self.rng.shuffle(self.chunks)
            split_idx = int(len(self.chunks) * split_ratio)
            
            if split == 'train':
                self.chunks = self.chunks[:split_idx]
            else:
                self.chunks = self.chunks[split_idx:]
            
    def __iter__(self):
        # Shuffle chunks if requested
        if self.shuffle:
            self.rng.shuffle(self.chunks)
            
        # Iterate through chunks
        for chunk_start, chunk_end in self.chunks:
            indices = list(range(chunk_start, chunk_end))
            
            # Shuffle indices within chunk if requested
            if self.shuffle:
                self.rng.shuffle(indices)
                
            # Yield batches from this chunk
            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i + self.batch_size]
                
    def __len__(self):
        total_samples = sum(end - start for start, end in self.chunks)
        return (total_samples + self.batch_size - 1) // self.batch_size
        
        
            
class HDF3DIterableDataset(torch.utils.data.IterableDataset):
    """Iterable dataset that distributes chunks across workers for HDF5 data."""
    
    def __init__(self, file_path, dataset_name, transform=None, 
                 chunk_size=1000, adaptive_chunk_size_workers=None, shuffle=True, seed=None):
        """Initialize the dataset.
        
        Args:
            file_path (str): Path to the HDF5 file
            dataset_name (str): Name of the dataset within the HDF5 file
            transform: Optional transform to apply to data
            chunk_size (int): Size of chunks to process
            shuffle (bool): Whether to shuffle chunks
            seed (int): Random seed for shuffling
        """
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.transform = transform
        self.adaptive_chunk_size_workers = adaptive_chunk_size_workers
        self.shuffle = shuffle
        self.seed = seed
        self.iter_idx = 0
        
        # Get dataset shape
        with h5py.File(file_path, 'r') as f:
            self.shape = f[dataset_name].shape
            if len(self.shape) != 3:
                raise ValueError("Dataset must be 3-dimensional")
            
        if not isinstance(self.adaptive_chunk_size_workers, int):
            self.chunk_size = chunk_size
        else:
            #calculate the optimal chunk size for the training set
            #the chunk size should be a multiple of the row size
            #and should evenly divide the number of training samples
            #into the number of workers
            self.chunk_size = (self.shape[0] // self.adaptive_chunk_size_workers) * self.shape[1]
            
            #make sure the chunk size is not larger than the dataset
            #and not larger than the maximum chunk size
            self.chunk_size = min(min(self.chunk_size, chunk_size), self.shape[0] * self.shape[1])

        
        # Calculate chunk boundaries
        self.chunks = []
        pos = 0
        while pos < self.shape[0] * self.shape[1]:
            chunk_start = pos
            # Round chunk_end up to end of row
            row_end = ceil((chunk_start + self.chunk_size) / self.shape[1])
            chunk_end = min(self.shape[0] * self.shape[1], row_end * self.shape[1])
            self.chunks.append((chunk_start, chunk_end))
            pos = chunk_end
            
        self.chunk_size = self.chunks[0][1] - self.chunks[0][0]
        
        self.hdf_file = None
        self.rng = None
    
    def _get_worker_chunks(self):
        """Get chunks assigned to this worker."""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:  # Single-process data loading
            return range(len(self.chunks))
            
        # Divide chunks between workers
        per_worker = int(ceil(len(self.chunks) / worker_info.num_workers))
        worker_id = worker_info.id
        start_idx = worker_id * per_worker
        end_idx = min(start_idx + per_worker, len(self.chunks))
        
        return range(start_idx, end_idx)
    
    def __iter__(self):
        """Return iterator over assigned chunks."""
        # Initialize random state
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        self.rng = np.random.default_rng(self.seed*100+self.iter_idx*10 + worker_id if self.seed is not None else None)
        self.iter_idx += 1
        self.open()
        
        # Shuffle chunks if requested
        if self.shuffle:
            self.rng.shuffle(self.worker_chunks)
        
        # Iterate through assigned chunks
        for chunk_idx in self.worker_chunks:
            chunk_start, chunk_end = self.chunks[chunk_idx]
            
            # Convert to 2D coordinates
            start_i = chunk_start // self.shape[1]
            end_i = chunk_end // self.shape[1]
            
            # Load chunk
            chunk_data = self.hdf_file[self.dataset_name][start_i:end_i, :, :]
            
            # Generate indices within chunk
            indices = np.arange(chunk_end - chunk_start)
            if self.shuffle:
                self.rng.shuffle(indices)
            
            # Yield items from chunk
            for idx in indices:
                rel_i = idx // self.shape[1]
                rel_j = idx % self.shape[1]
                x = chunk_data[rel_i, rel_j, :]
                
                if self.transform is not None:
                    x = self.transform(x)
                yield x
    
    def __len__(self):
        """Return total length of dataset."""
        return self.shape[0] * self.shape[1]
    
    def close(self):
        """Close the HDF5 file."""
        if self.hdf_file is not None:
            self.hdf_file.close()
            self.hdf_file = None

        self.worker_chunks = None
            
    def open(self):
        """Open the HDF5 file."""
        self.close()
        # Open file
        self.hdf_file = h5py.File(self.file_path, 'r')
        
        # Get chunks for this worker
        self.worker_chunks = list(self._get_worker_chunks())

    def worker_init_fn(self, worker_id):
        """Initialize worker."""
        self._worker_id = worker_id
        self.open()

    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()