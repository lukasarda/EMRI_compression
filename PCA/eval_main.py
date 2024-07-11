import numpy as np
import cupy as cp
import torch
from dataset_class import npz_class
from torch.utils.data import DataLoader
from helpers import mat_overlap, traf_inv_traf


def flatten_and_concatenate(batch, device):
    original_shape = batch[0].shape    
    flattened_batch = [item.view(item.shape[0], -1) for item in batch]
    data_matrix = cp.array(torch.cat(flattened_batch, dim=0))
    data_matrix_torch = torch.tensor(cp.asnumpy(data_matrix)).to(device)
    
    return data_matrix_torch, original_shape

def reshape_processed_data(processed_data, original_shape):
    processed_data_cpu = processed_data.clone().detach().cpu() 
    num_elements = original_shape[0]
    
    # Split the processed data tensor
    split_tensors = torch.split(processed_data_cpu, num_elements)
    
    # Reshape each split tensor to its original shape
    reshaped_data = [tensor.view(original_shape) for tensor in split_tensors]
    
    # Convert to CuPy arrays
    reshaped_data = [cp.array(tensor.numpy()) for tensor in reshaped_data]
    
    return reshaped_data



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    traf_matrix_path = '/sps/lisaf/lkarda/H_matrices_td/30000_samples_1_months/ipca_1000_components.npz'
    data_folder = '/sps/lisaf/lkarda/H_matrices_td/100_samples_1_months/tfm_singles'
    dataset = npz_class(dirpath=data_folder)

    file = np.load(traf_matrix_path)
    traf_matrix = torch.tensor(cp.asnumpy(file[file.files[2]])).to(device)

    test_loader = DataLoader(dataset,
                              batch_size= 1,
                              shuffle= True,
                              num_workers= 1)

    eval_list= list(np.arange(0, 1001, 1))
    
    for i, batch in enumerate(test_loader):
        for n in eval_list:
            # Move batch to device
            batch = [item.to(device) for item in batch]
            data_matrix, original_shape = flatten_and_concatenate(batch, device)
            
            print(traf_matrix[:n].shape)
            processed_data_matrix, data_matrix = traf_inv_traf(data_matrix, traf_matrix[:n])
            processed_batch = reshape_processed_data(processed_data_matrix, original_shape)

            # processed_batch_np = np.array([pb.get() for pb in processed_batch])
            # batch_np = np.array([b.cpu().numpy() for b in batch])
            # np.savez_compressed('/sps/lisaf/lkarda/test_bin/comp_in_out_PCA_998th_to_{}_component.npz'.format(n), output= processed_batch_np, input= batch_np)

            batch_cp = cp.array(torch.cat(batch, dim=0).cpu().numpy())
            processed_batch_cp = cp.array(torch.cat([torch.tensor(cp.asnumpy(pb)).to(device) for pb in processed_batch], dim=0).cpu().numpy())

            overlap= mat_overlap(batch_cp, processed_batch_cp)
            print(overlap)

            print(n)

        if i == 0:
            print(i, 'reached')
            break


if __name__ == "__main__":
    main()
