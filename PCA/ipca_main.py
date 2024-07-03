import cupy as cp
import numpy as np
import torch
from torch.utils.data import DataLoader
from cuml.decomposition import IncrementalPCA

from dataset_class import npz_class
# from helpers import mat_traf_inv_traf

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    folder = '30000_samples_1_months/'
    dir_path = '/sps/lisaf/lkarda/H_matrices_td/' + folder 
    test_path = dir_path + 'tfm_singles/'
    dataset = npz_class(dirpath=test_path)

    # vali_path = test_path + 'tfm_singles_validation'
    # vali_set = npz_class(dirpath=vali_path)
    # vali_loader = DataLoader(vali_set, batch_size=20, shuffle=True, num_workers=num_workers)

    batch_size = 500
    num_workers = 4

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


    ipca = IncrementalPCA()

    
    for i, batch in enumerate(train_loader):
        ### Training
        batch = [item.to(device) for item in batch]
        # Flatten the last two dimensions
        flattened_batch = [item.view(item.shape[0], -1) for item in batch]
        # Concatenate the flattened items along the first dimension
        data_matrix = cp.array(torch.cat(flattened_batch, dim=0))

        ipca.partial_fit(data_matrix)

        print(i+1, '/', len(train_loader))
        print('Components:', ipca.n_components_)
        print('Samples seen:', ipca.n_samples_seen_)

        # ### Validation
        # if (i+1) % 4 == 0:
        #     val_overlap_sum = 0
        #     val_batches = 0
            
        #     for val_batch in vali_loader:
        #         val_batch = [item.to(device) for item in val_batch]
        #         val_flattened_batch = [item.view(item.shape[0], -1) for item in batch]
        #         val_data_matrix = cp.array(torch.cat(val_flattened_batch, dim=0))
                
        #         overlap = mat_traf_inv_traf(ipca, val_data_matrix)
        #         val_overlap_sum += overlap
        #         val_batches += 1

        #     average_overlap = val_overlap_sum / val_batches
        #     print(f'Validation overlap at batch {i}: {average_overlap}')

    print('Final components:', ipca.n_components_)
    print('Final samples seen:', ipca.n_samples_seen_)

    print(ipca.components_.shape)
    print(ipca.singular_values_.shape)

    ### Save model
    np.savez_compressed(
        dir_path + 'ipca_{}_components'.format(2*batch_size),
        n_components= ipca.n_components_,
        n_samples_seen= ipca.n_samples_seen_,
        components= ipca.components_,
        singular_values= ipca.singular_values_
        )


if __name__ == "__main__":
    main()