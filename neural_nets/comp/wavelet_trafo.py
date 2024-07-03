import os
import numpy as np
from ltft.timeserie import Timeserie
from ltft.tfmap import TimeFrequencyArray
from ltft.common.utils import select_scale_df

import matplotlib.pyplot as plt

def read_global_extremes(file_path):
    """Read global maximum and minimum values from a text file."""
    global_max = None
    global_min = None
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('Global Max'):
                global_max = float(line.split(': ')[1])
            elif line.startswith('Global Min'):
                global_min = float(line.split(': ')[1])
    
    return global_min, global_max

def load_file(file):
    """Load the signal from file and p"""
    sample_id = os.path.splitext(os.path.basename(file))[0]
    signal = np.load(file)
    signal = signal[signal.files[0]][0]
    return sample_id, signal

def normalization(arr, arr_min, arr_max, up=1, low=0):
    """Normalize array values to a given range."""
    scale = arr_max - arr_min
    alpha = (up - low) / scale
    beta = (low * arr_max - up * arr_min) / scale
    normed_arr = alpha * arr + beta
    return np.clip(normed_arr, low, up)

def process_signal(signal, dt, fs, window, res_factor):
    """Process the signal to get the real and imaginary parts of its wavelet transform."""
    ts = Timeserie(signal, dt=dt)
    target_df = 1.e-4  # Hz
    scale = int(select_scale_df(target_df, fs) * res_factor)
    
    return ts.dwilt(scale, window, q=32)

def process_file(signal, dt, fs, window, res_factor):
    """Process its real and imaginary parts."""
    
    tfm_real = process_signal(signal.real, dt, fs, window, res_factor)
    tfm_imag = process_signal(signal.imag, dt, fs, window, res_factor)

    return tfm_real, tfm_imag

def wavelet_transform(signal, global_min, global_max, dt=10., fs=1./10., window='meyer', res_factor=1.):
    """Perform wavelet transform and normalize the results."""
    tfm_real, tfm_imag = process_file(signal, dt, fs, window, res_factor)

    cfgs= [tfm_real.cfg, tfm_imag.cfg]
    
    tfm_real_normed = normalization(tfm_real.data.transpose(), arr_min=global_min, arr_max=global_max)
    tfm_imag_normed = normalization(tfm_imag.data.transpose(), arr_min=global_min, arr_max=global_max)
    
    tfms= np.array([tfm_real_normed, tfm_imag_normed])
    return tfms, cfgs

def inv_process_signal(signal, fs, window, res_factor, cfg):
    """Invert Process the signal to get the timeseries from the wavelet representation."""
    tfm = TimeFrequencyArray(signal, fs=fs)
    tfm.cfg= cfg
    target_df = 1.e-4  # Hz
    scale = int(select_scale_df(target_df, fs) * res_factor)

    return tfm.idwilt(scale, window, q=32)

def inverse_process_file(tfm, dt, fs, window, res_factor, cfgs):
    """Process its real and imaginary parts."""
    
    ts_real = inv_process_signal(tfm[0], fs, window, res_factor, cfgs[0])
    ts_imag = inv_process_signal(tfm[1], fs, window, res_factor, cfgs[1])

    return ts_real, ts_imag

def inverse_normalization(normed_arr, arr_min, arr_max, up=1, low=0):
    """Invert the normalization of array values to their original range."""
    scale = arr_max - arr_min
    alpha = (up - low) / scale
    beta = (low * arr_max - up * arr_min) / scale
    inv_arr = (normed_arr - beta) / alpha
    return np.clip(inv_arr, arr_min, arr_max)

def inv_wavelet_transform(output, global_min, global_max, cfgs, dt=10., fs=1./10., window='meyer', res_factor=1.):
    """Perform inverse normalization and inverse wavelet transform."""
    tfm_real = inverse_normalization(output[0], arr_min=global_min, arr_max=global_max)
    tfm_imag = inverse_normalization(output[1], arr_min=global_min, arr_max=global_max)

    tfm= np.array([tfm_real.transpose(), tfm_imag.transpose()])
    signal_real, signal_imag = inverse_process_file(tfm, dt, fs, window, res_factor, cfgs)

    return np.array([signal_real, signal_imag])
