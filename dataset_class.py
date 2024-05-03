import torch
import os
import numpy as np

from torch.utils.data import Dataset, ConcatDataset


class Dataset:
    
    def __init__(self, directory):
        self.directory = directory
        self.data_files = [filename for filename in sorted(os.listdir(directory)) if filename.endswith(".pt")]
        self.total_samples = sum(self._get_samples_count(file_path) for file_path in self.data_files)

    def _get_samples_count(self, file_path):
        return torch.load(os.path.join(self.directory, file_path)).size(0)

    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, index):
        # Find the corresponding file
        current_index = 0
        for file_path in self.data_files:
            samples_count = self._get_samples_count(file_path)
            if index < current_index + samples_count:
                # Load the tensor from the file
                tensor = torch.load(os.path.join(self.directory, file_path))
                # Return the specific sample
                return tensor[index - current_index]
            current_index += samples_count
        raise IndexError("Index out of range")


class npz_dataset(Dataset):

    def __init__(self, filepath, key):
        self.filepath = filepath
        self.key = key


    def __getitem__(self, index):
        #matrix = np.load(self.filepath)[self.key]
        matrix_mmap = np.load(self.filepath, mmap_mode='r')
        element = matrix_mmap[self.key][index]
        matrix_mmap.close()
        return element #matrix[index]
    
    def __len__(self):
        matrix_mmap = np.load(self.filepath, mmap_mode='r')
        length = matrix_mmap[self.key].shape[0]
        matrix_mmap.close()
        return length