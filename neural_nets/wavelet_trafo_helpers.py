import numpy as np
import os
from ltft.timeserie import Timeserie
from ltft.common.utils import select_scale_df

def wavelet_trafo(window, signal, dt, fs, res_factor=1.):
    ts = Timeserie(signal, dt=dt)
    target_df = 1.e-4  # Hz
    s = int(select_scale_df(target_df, fs) * res_factor)
    return ts.dwilt(s, window, q=32)

def normalization(arr, arr_min, arr_max, up, low):
    scale = (arr_max - arr_min)
    alpha = (up - low) / scale
    beta = (low * arr_max - up * arr_min) / scale
    normed_arr = alpha * arr + beta
    return np.clip(normed_arr, low, up)

def process_file(file, dt, fs, window, res_factor, key):
    sample_id = os.path.splitext(os.path.basename(file))[0]
    signal = np.load(file)[key]
    tfm_real = wavelet_trafo(window, signal.real.squeeze(), dt, fs, res_factor=res_factor)
    tfm_imag = wavelet_trafo(window, signal.imag.squeeze(), dt, fs, res_factor=res_factor)
    return sample_id, tfm_real, tfm_imag

def calculate_global_extremes(temp_dir, file_list, dt, fs, window, res_factor, key):
    global_max = float('-inf')
    global_min = float('inf')

    for i, file in enumerate(file_list):
        sample_id, tfm_real, tfm_imag = process_file(file, dt, fs, window, res_factor, key)
        
        # Save intermediate transforms to temporary files
        np.savez_compressed(os.path.join(temp_dir, sample_id + '_tfm.npz'), real=tfm_real.data, imag=tfm_imag.data)
        
        max_real = np.max(tfm_real.data)
        min_real = np.min(tfm_real.data)
        max_imag = np.max(tfm_imag.data)
        min_imag = np.min(tfm_imag.data)

        global_max = max(global_max, max_real, max_imag)
        global_min = min(global_min, min_real, min_imag)

        print(f'Calculating extremes: {i+1}/{len(file_list)}')

    return global_max, global_min

def write_global_extremes(data_dir, global_max, global_min):
    with open(os.path.join(data_dir, 'global_extremes.txt'), 'w') as f:
        f.write(f'Global Max: {global_max}\n')
        f.write(f'Global Min: {global_min}\n')