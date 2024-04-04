import numpy as np
import os
import re
import torch
from numba import njit



def fill_length_zeros(array, length):
    right_length = np.zeros(length)
    right_length[:len(array)] = array
    return right_length

def find_length(hf_list):
    return max(map(len, hf_list))



### Transform array of complex numbers into array of reals, by concatenating real and imaginary parts
### Applicable to time and frequency domain

#@njit(parallel=True)
def conca(h):
    stacked_array = np.stack((h.real, h.imag), axis=-1)
    return stacked_array.reshape(stacked_array.shape[0], -1)


def file_paths_conctructor(path):
    
    # Get files in path
    a = os.listdir(path)
    
    # Sort them by sample order
    pattern = re.compile(r'^\d+')
    files = sorted(a, key=lambda x: int(re.match(pattern, x).group()))

    file_paths = list(map(lambda x: path + x, files))

    return file_paths



### Key is either ht, hf or hw corresponding to time, frequency or wavelet domain

@njit(parallel=True)
def apply_function_to_npz_files(file_paths, func, key):
    matrices = []

    for file_path in file_paths:
        # Load the .npz file
        data = np.load(file_path)

        # Apply the function to the loaded data
        result_matrix = func(data[key])
        matrices.append(result_matrix)

    return np.concatenate(matrices, axis=0)



### Compute the overlaps of two sets of arrays

def overlap(a, b):
    x = np.array(a)
    y = np.array(b)
    return np.abs(np.sum(x * y, axis=1)/(np.linalg.norm(x, axis=1)*np.linalg.norm(y, axis=1)))


### Compute the overlaps of initial data and retransformed data with a given (i)PCA model

#@njit(parallel=True)
def traf_inv_traf(pca_type, x):
    x_traf = pca_type.transform(x)
    x_inv_traf = pca_type.inverse_transform(x_traf)
    return overlap(x, x_inv_traf)



### Data processing: Transform numpy .npz to pytorch .pt files
### Key is either ht, hf or hw corresponding to time, frequency or wavelet domain

def npz_to_tensor(npz_file, output_file, key):
    # Load the .npz file
    data = np.load(npz_file)
    
    # Extract the array under the 'hf' keyword
    array = np.array(conca(data[key]))
    
    # Convert the NumPy array to a PyTorch tensor
    tensor = torch.tensor(array)
    
    # Save the PyTorch tensor
    torch.save(tensor, output_file)

def npz_directory_to_tensor(directory, key):
    # Ensure the output directory exists
    output_directory = os.path.join(directory, 'tensor_files')
    os.makedirs(output_directory, exist_ok=True)
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".npz"):
            npz_file = os.path.join(directory, filename)
            output_file = os.path.join(output_directory, os.path.splitext(filename)[0] + '.pt')
            npz_to_tensor(npz_file, output_file, key)

# Example usage:
# npz_directory_to_tensor('./H_matrices_fd/100_samples', 'hf')