import numpy as np
from numba import njit

@njit


def fill_length_zeros(array, length):
    right_length = np.zeros(length)
    right_length[:len(array)] = array
    return right_length

def find_length(hf_list):
    return max(map(len, hf_list))



### Applicable to time and frequency domain

def conca(h):
    stacked_array = np.stack((h.real, h.imag), axis=-1)
    return stacked_array.reshape(stacked_array.shape[0], -1)


