import cupy as cp
import numpy as np
import torch
from torch.utils.data import DataLoader
from cuml.decomposition import IncrementalPCA
from dataset_class import npz_class

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    folder = '30000_samples_1_months/'
    dir_path = '/sps/lisaf/lkarda/H_matrices_td/' + folder 
    test_path = dir_path + 'tfm_singles/'
    dataset = npz_class(dirpath=test_path)

    batch_size = 500
    num_workers = 4

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    ipca = IncrementalPCA()

    ### Training
    for i, batch in enumerate(train_loader):
        batch = [item.to(device) for item in batch]
        flattened_batch = [item.view(item.shape[0], -1) for item in batch]
        data_matrix = cp.array(torch.cat(flattened_batch, dim=0))

        ipca.partial_fit(data_matrix)

        print(i+1, '/', len(train_loader))
        print('Components:', ipca.n_components_)
        print('Samples seen:', ipca.n_samples_seen_)

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
        singular_values= ipca.singular_values_,
        explained_variance= ipca.explained_variance_,
        explained_variance_ratio_= ipca.explained_variance_ratio_
        )


if __name__ == "__main__":
    main()