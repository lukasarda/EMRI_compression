import time
start_time=time.time()

import os
import numpy as np
from dataset_class import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from cuml.decomposition import IncrementalPCA

from helpers import traf_inv_traf



data_dir = '/sps/lisaf/lkarda/H_matrices_td/5020_samples/tensor_files'
dataset = Dataset(data_dir)

batch_size = 4  # Specify the batch size
threshold = 0.97   # Threshold for overlap - exit condition for training the model - in [0,1]
get_projection_matr = False #If True, saves the projection matrix and singular values of each fit step to NumPy files



print(data_dir, batch_size, threshold)

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
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=True)


ipca = IncrementalPCA(n_components=batch_size)

""" 
for batch in train_loader:

    # Train model
    ipca.partial_fit(np.array(batch))
    

overlaps = []
for batch in test_loader:

    # Test model
    overlaps.append(traf_inv_traf(ipca, np.array(batch)))

 """


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
        print(np.average(overlaps))

        if get_projection_matr == True:
            os.makedirs(data_dir + '/p_matrix_svs/', exist_ok=True)
            np.savez_compressed(data_dir + '/p_matrix_svs/' + str(i*batch_size), p_matrix=ipca.components_, svs=ipca.singular_values_)
            break
else:
    print(i*batch_size, '/', train_size, 'batches used')


print(np.average(overlaps))



end_time=time.time()
runtime=end_time-start_time
print('Runtime:', runtime, 'seconds')