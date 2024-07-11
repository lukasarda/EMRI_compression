import numpy as np
import cupy as cp
from ltft.timeserie import Timeserie
from ltft.tfmap import TimeFrequencyArray
from ltft.common.utils import select_scale_df


def overlap(a, b):
    x = np.array(a)
    y = np.array(b)
    return np.abs(np.sum(x * y, axis=1)/(np.linalg.norm(x, axis=1)*np.linalg.norm(y, axis=1)))

def mat_overlap(a,b):
    nom= cp.multiply(a,b).sum()
    denom= cp.linalg.norm(a) * cp.linalg.norm(b)
    return nom/denom

def traf_inv_traf(data, traf):
    trafo= data@traf.T
    inv_trafo= trafo@traf
    return data, inv_trafo

def pca_traf_inv_traf(pca_type, x):
    x_traf = pca_type.transform(x)
    x_inv_traf = pca_type.inverse_transform(x_traf)
    return overlap(x, x_inv_traf)


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



def inv_process_signal(signal, fs, window, res_factor, cfg):
    """Invert Process the signal to get the timeseries from the wavelet representation."""
    tfm = TimeFrequencyArray(signal, fs=fs)
    if cfg is not None:
        tfm.cfg = cfg
    target_df = 1.e-4  # Hz
    scale = int(select_scale_df(target_df, fs) * res_factor)

    return tfm.idwilt(scale, window, q=32)

def inverse_process_file(tfm, dt, fs, window, res_factor, cfgs):
    """Process its real and imaginary parts."""
    ts_real = inv_process_signal(tfm[0], fs, window, res_factor, cfgs[0] if cfgs else None)
    ts_imag = inv_process_signal(tfm[1], fs, window, res_factor, cfgs[1] if cfgs else None)

    return ts_real, ts_imag

def inverse_normalization(normed_arr, arr_min, arr_max, up=1, low=0):
    """Invert the normalization of array values to their original range."""
    scale = arr_max - arr_min
    alpha = (up - low) / scale
    beta = (low * arr_max - up * arr_min) / scale
    inv_arr = (normed_arr - beta) / alpha
    return np.clip(inv_arr, arr_min, arr_max)

def inv_wavelet_transform(output, global_min, global_max, cfgs=None, dt=10., fs=1./10., window='meyer', res_factor=1.):
    """Perform inverse normalization and inverse wavelet transform."""
    tfm_real = inverse_normalization(output[0], arr_min=global_min, arr_max=global_max)
    tfm_imag = inverse_normalization(output[1], arr_min=global_min, arr_max=global_max)

    tfm = np.array([tfm_real.transpose(), tfm_imag.transpose()])
    signal_real, signal_imag = inverse_process_file(tfm, dt, fs, window, res_factor, cfgs)

    return np.array([signal_real.data, signal_imag.data])




# import numpy as np
# a = np.load('./9_9_sample.npz')
# len(a[a.files[0]])