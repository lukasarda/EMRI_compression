import time

start_time=time.time()


import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from cuml.decomposition import IncrementalPCA

from helpers import traf_inv_traf



class ConcatenatedDataset(Dataset):
    def __init__(self, directory):
        self.data = []
        self.total_samples = 0
        
        # Iterate over files in the directory
        for filename in sorted(os.listdir(directory)):
            if filename.endswith(".pt"):
                file_path = os.path.join(directory, filename)
                # Load tensor from each .pt file
                tensor = torch.load(file_path)
                # Append tensor to the data list
                self.data.append(tensor)
                # Update total number of samples
                self.total_samples += len(tensor)

    def __getitem__(self, index):
        # Find the file index and sample index within the file
        file_index = 0
        sample_index = index
        for tensor in self.data:
            if sample_index < len(tensor):
                break
            sample_index -= len(tensor)
            file_index += 1
        return self.data[file_index][sample_index]
    
    def __len__(self):
        return self.total_samples


# Example usage:
data_dir = '/sps/lisaf/lkarda/H_matrices_fd/100_samples/tensor_files'
dataset = ConcatenatedDataset(data_dir)


batch_size = 10  # Specify the batch size
threshold = 0.01
get_projection_matr = True


# Define the size of the training and testing sets
train_size = int(0.8 * len(dataset))  # 80% of the data for training
test_size = len(dataset) - train_size  # Remaining 20% for testing

# Define indices for the training and testing sets
indices = list(range(len(dataset)))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

# Create data samplers for training and testing sets
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Create data loaders for training and testing sets
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=1)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=1)


ipca = IncrementalPCA(n_components=batch_size)

i=0
for batch in train_loader:
    i+=1

    # Train model
    ipca.partial_fit(np.array(batch))
    
    overlaps = []
    for batch in test_loader:

        # Test model
        overlaps.append(traf_inv_traf(ipca, np.array(batch)))
    if np.average(overlaps) >= threshold:
        print(i*batch_size, '/', train_size, 'batches used')

        if get_projection_matr == True:
            os.makedirs(data_dir + '/projection_matrix/', exist_ok=True)
            os.makedirs(data_dir + '/singular_values/', exist_ok=True)
            np.savez_compressed(data_dir + '/projection_matrix/' + str(i*batch_size), ipca.components_)
            np.savez_compressed(data_dir + '/singular_values/' + str(i*batch_size), ipca.singular_values_)
        break
else:
    print(i*batch_size, '/', train_size, 'batches used')



print(np.average(overlaps))
print(ipca.components_.shape)
print(ipca.singular_values_.shape)


end_time=time.time()

runtime=end_time-start_time
print('Runtime:', runtime, 'seconds')