import torch
import os
import numpy as np

from torch.utils.data import Dataset, ConcatDataset



class npz_class(Dataset):

    def __init__(self, dirpath):
        self.filepaths = [os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.endswith('.npz')]

    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        filepath= self.filepaths[idx]

        with np.load(filepath) as ts:
            real = ts[ts.files[0]].real
            imag = ts[ts.files[0]].imag

        # Combine the arrays
        combi = np.array([real, imag])
        return combi