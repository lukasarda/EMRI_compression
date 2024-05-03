import time
start_time=time.time()

import os
import glob
import torch
import numpy as np

from torch.utils.data import ConcatDataset, DataLoader, SubsetRandomSampler
from dataset_class import npz_dataset
#from sklearn.decomposition import IncrementalPCA
from cuml.decomposition import IncrementalPCA

from helpers import traf_inv_traf, conca


folder = '30000_samples'
dir_path = '/sps/lisaf/lkarda/H_matrices_fd/'+folder+'/'
key = 'hf'

filepaths = glob.glob(os.path.join(dir_path, '*.npz'))
datasets = [npz_dataset(filepath=filepath, key=key) for filepath in filepaths]
big_set = ConcatDataset(datasets)




if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')


    batch_size = 20  # Specify the batch size
    threshold = 0.97   # Threshold for overlap - exit condition for training the model - in [0,1]
    get_projection_matr = False #If True, saves the projection matrix and singular values of each fit step to NumPy files

    num_workers = 4

    print('Dataset:', dir_path)
    print('Batch size:', batch_size)
    print('Threshold:', threshold)

    # Define the size of the training and testing sets
    train_size = int(0.8 * len(big_set))  # 80% of the data for training
    test_size = len(big_set) - train_size  # Remaining 20% for testing

    # Define indices for the training and testing sets
    indices = list(range(len(big_set)))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Create data samplers for training and testing sets
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create data loaders for training and testing sets
    train_loader = DataLoader(big_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(big_set, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=True)


    ipca = IncrementalPCA(n_components=batch_size)

    init_time = time.time()
    init = init_time-start_time
    print('Initialisation time:', init, 'seconds')

    for i, train_batch in enumerate(train_loader):


        matrix = np.stack([conca(item).ravel() for item in train_batch], axis=0)
        # Train model
        ipca.partial_fit(matrix)
        
        if i % 10 ==0:
            overlaps = []
            for test_batch in test_loader:

                matrix = np.stack([conca(item).ravel() for item in test_batch], axis=0)
                # Test model
                overlaps.append(np.average(traf_inv_traf(ipca, matrix)))

            print(f"Batch {i+1}/{len(train_loader)}: Average overlap = {overlaps[-1]}")
            np.savez_compressed('/sps/lisaf/lkarda/overlaps/'+str(len(big_set))+'_'+str(batch_size)+'.npz', overlaps=overlaps)

            
            if overlaps[-1] >= threshold:
                print(i*batch_size, '/', train_size, 'batches used')
                print(overlaps[-1])

            if get_projection_matr == True:
                os.makedirs(dir_path + '/p_matrix_svs/', exist_ok=True)
                np.savez_compressed(dir_path + '/p_matrix_svs/' + str(i*batch_size), p_matrix=ipca.components_, svs=ipca.singular_values_)
                break
    else:
        print(i+1, '/', len(train_loader), 'training batches used')
        print('Final overlap:', overlaps[-1])

    
    
    
    
    
    
    end_time=time.time()
    train_time=end_time-init_time
    print('Traintime:', train_time, 'seconds')
    runtime=end_time-start_time
    print('Runtime:', runtime, 'seconds')
