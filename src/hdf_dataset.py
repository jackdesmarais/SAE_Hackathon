import h5py
import numpy as np

class HDF3DIterator:
    """Iterator class for accessing first two dimensions of a 3D HDF5 dataset.
    
    This class provides methods to iterate through or index into the first two dimensions
    of a 3D array stored in an HDF5 file, while keeping the third dimension intact.
    
    Attributes:
        file_path (str): Path to the HDF5 file
        dataset_name (str): Name of the dataset within the HDF5 file
        shape (tuple): Shape of the 3D array
        current_idx (int): Current position in iteration
    """
    
    def __init__(self, file_path, dataset_name, transform=None, preload=False):
        """Initialize the iterator.
        
        Args:
            file_path (str): Path to the HDF5 file
            dataset_name (str): Name of the dataset within the HDF5 file
        """
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.transform = transform
        with h5py.File(file_path, 'r') as f:
            self.shape = f[dataset_name].shape
            if len(self.shape) != 3:
                raise ValueError("Dataset must be 3-dimensional")

        self.hdf_file=None
        self.current_idx = 0
        self.preload = preload
        self.data = None

    def __len__(self):
        """Return the total number of items in first two dimensions."""
        return self.shape[0] * self.shape[1]
        
    def __getitem__(self, idx):
        """Get item at specified index.
        
        Args:
            idx (tuple or int): Index into first two dimensions. Can be:
                - slice object
                - list or numpy array of indices
                - tuple of (i,j) coordinates
                - flat index that will be converted to (i,j)
                
        Returns:
            numpy.ndarray: 1D array corresponding to data at index
        """
        if isinstance(idx, slice):
            # Handle slice object
            start = 0 if idx.start is None else idx.start
            stop = len(self) if idx.stop is None else idx.stop
            step = 1 if idx.step is None else idx.step
            
            results = [self[i] for i in range(start, stop, step)]
            return np.stack(results)
        elif isinstance(idx, list) or isinstance(idx, np.ndarray):
            results = [self[i] for i in idx]
            return np.stack(results)
        else:
            if isinstance(idx, int):
                if idx >= len(self):
                    raise IndexError("Index out of bounds")
                # Convert flat index to 2D coordinates
                i = idx // self.shape[1]
                j = idx % self.shape[1]
            elif isinstance(idx, tuple) and len(idx) == 2:
                i, j = idx
                if i >= self.shape[0] or j >= self.shape[1]:
                    raise IndexError("Index out of bounds") 
            else:
                raise ValueError("Index must be, list, array, integer or 2-tuple")
                
            if self.data is not None:
                x= self.data[i,j,:]
            elif self.hdf_file is not None:
                x= self.hdf_file[self.dataset_name][i,j,:]
            else:
                raise ValueError("HDF5 file not opened")
            
            if self.transform is not None:
                x = self.transform(x)
            return x
            
    def __iter__(self):
        """Initialize iterator."""
        self.current_idx = 0
        return self
        
    def __next__(self):
        """Get next item in iteration.
        
        Returns:
            numpy.ndarray: Next 1D array in sequence
            
        Raises:
            StopIteration: When iteration is complete
        """
        if self.current_idx >= len(self):
            raise StopIteration
            
        item = self[self.current_idx]
        self.current_idx += 1
        return item

    def open(self):
        self.hdf_file = h5py.File(self.file_path, 'r')
        print(f'hdf_file - opened')
        if self.preload:
            print(f'hdf_file - preloading')
            self.data = self.hdf_file[self.dataset_name][:,:,:]
            print(f'hdf_file - preloaded')

    def close(self):
        self.hdf_file.close()
        self.hdf_file=None
        self.data=None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

        