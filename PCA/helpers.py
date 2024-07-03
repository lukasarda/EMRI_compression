import numpy as np
import cupy as cp
import os
import re
import torch
from numba import njit


### Compute the overlaps of two sets of arrays

def overlap(a, b):
    x = np.array(a)
    y = np.array(b)
    return np.abs(np.sum(x * y, axis=1)/(np.linalg.norm(x, axis=1)*np.linalg.norm(y, axis=1)))

def mat_overlap(a,b):
    nom= cp.multiply(a,b).sum()
    denom= cp.linalg.norm(a) * cp.linalg.norm(b)
    return nom/denom

### Compute the overlaps of initial data and retransformed data with a given (i)PCA model

#@njit(parallel=True)
def traf_inv_traf(pca_type, x):
    x_traf = pca_type.transform(x)
    x_inv_traf = pca_type.inverse_transform(x_traf)
    return overlap(x, x_inv_traf)

def mat_traf_inv_traf(pca_type, x):
    x_traf = pca_type.transform(x)
    x_inv_traf = pca_type.inverse_transform(x_traf)
    return mat_overlap(x, x_inv_traf)


# Example usage:
# npz_directory_to_tensor('./H_matrices_fd/100_samples', 'hf')

# import numpy as np
# a = np.load('./9_9_sample.npz')
# len(a[a.files[0]])