import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import torch
from dataset_class import npz_class
from torch.utils.data import DataLoader


def traf_inv_traf(data, traf):
    trafo= data@traf.T
    inv_trafo= trafo@traf
    return data, inv_trafo

def mat_overlap(a,b):
    nom= cp.multiply(a,b).sum()
    denom= cp.linalg.norm(a) * cp.linalg.norm(b)
    return nom/denom

def plot_all_overlaps(n_components, overlaps, output_path):
    plt.scatter(n_components, overlaps)
    plt.xlabel('# of components')
    plt.ylabel('Overlap')
    plt.xlim(0., len(overlaps))
    plt.ylim(0., 1.1)

    plt.savefig(output_path)

def flatten_and_concatenate(batch, device):
    original_shape = batch[0].shape    
    flattened_batch = [item.view(item.shape[0], -1) for item in batch]
    data_matrix = cp.array(torch.cat(flattened_batch, dim=0))
    data_matrix_torch = torch.tensor(cp.asnumpy(data_matrix)).to(device)
    
    return data_matrix_torch, original_shape

def reshape_processed_data(processed_data, original_shape):
    processed_data_cpu = processed_data.clone().detach().cpu() 

    # Calculate the total number of elements in each item
    num_elements = original_shape[0]
    
    # Split the processed data tensor
    split_tensors = torch.split(processed_data_cpu, num_elements)
    
    # Reshape each split tensor to its original shape
    reshaped_data = [tensor.view(original_shape) for tensor in split_tensors]
    
    # Convert to CuPy arrays
    reshaped_data = [cp.array(tensor.numpy()) for tensor in reshaped_data]
    
    return reshaped_data

def process_batch(data_matrix):
    processed_data_matrix = data_matrix-1  # No processing in this placeholder
    return processed_data_matrix



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    traf_matrix_path = '/sps/lisaf/lkarda/H_matrices_td/30000_samples_1_months/ipca_1000_components.npz'
    data_folder = '/sps/lisaf/lkarda/H_matrices_td/100_samples_1_months/tfm_singles'

    file = np.load(traf_matrix_path)
    traf_matrix = torch.tensor(cp.asnumpy(file[file.files[2]])).to(device)

    dataset = npz_class(dirpath=data_folder)

    batch_size = 1
    num_workers = 1

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    eval_list= [999]


    
    for i, batch in enumerate(train_loader):
        for n in eval_list:
            # Move batch to device
            batch = [item.to(device) for item in batch]
            
            # Flatten and concatenate the batch
            data_matrix, original_shape = flatten_and_concatenate(batch, device)
            
            print(traf_matrix[998:n].shape)
            # Process the concatenated data
            processed_data_matrix, data_matrix = traf_inv_traf(data_matrix, traf_matrix[950:n])
            
            # Reshape the processed data back to original shapes
            processed_batch = reshape_processed_data(processed_data_matrix, original_shape)

            processed_batch_np = np.array([pb.get() for pb in processed_batch])
            batch_np = np.array([b.cpu().numpy() for b in batch])
            np.savez_compressed('/sps/lisaf/lkarda/test_bin/comp_in_out_PCA_998th_to_{}_component.npz'.format(n), output= processed_batch_np, input= batch_np)


            # Verify the shapes
            for processed in processed_batch:
                assert original_shape == processed.shape, f"Shape mismatch: {original_shape} != {processed.shape}"

            print(n)

        if i == 0:
            print(i, 'reached')
            break
            
    # batch_cp = cp.array(torch.cat(batch, dim=0).cpu().numpy())
    # processed_batch_cp = cp.array(torch.cat([torch.tensor(cp.asnumpy(pb)).to(device) for pb in processed_batch], dim=0).cpu().numpy())
    
    # print(mat_overlap(batch_cp[0], processed_batch_cp[0]))

    # print(f"Batch {i} overlap: {mat_overlap(batch_cp, processed_batch_cp)}")


if __name__ == "__main__":
    main()
