import numpy as np
import os
import glob


from ltft.timeserie import Timeserie
from ltft.common.utils import select_scale_df, get_df_bin, get_dt_bin
from ltft.tfr.lin.wilt import _get_nzeros_padding

from pywavelet.transforms import from_time_to_wavelet
from pywavelet.transforms.types import TimeSeries

from wavelet_trafo_helpers import wavelet_trafo, list_files

dt = 10. #s
fs = float(1./10.)
window = 'meyer'
res_factor = 1.

target_df = 1.e-4 # Hz


# dir_name= '5020_samples'
# data_dir = '/sps/lisaf/lkarda/H_matrices_td/'+dir_name+'/'

# file_list = list_files(data_dir)
#print(file_list)
file_list=['/sps/lisaf/lkarda/H_matrices_td/2_samples_1yr_td.npz']


for file in file_list[:1]:
    # signal = np.load(file)['ht'][0]
    signal = np.load(file)['H_trunc'][0]

    time = np.linspace(0., len(signal), len(signal)) * dt

    ### LTFT

    tf_map_real = wavelet_trafo(window, signal.real, dt, fs, res_factor=res_factor)
    #print(tf_map_real.data.shape)
    #tf_map_imag = wavelet_trafo(window, signal.imag, dt, fs, res_factor=res_factor)
    #print(tf_map_imag.shape)

    extents = [0., tf_map_real.dtbin*tf_map_real.ntbins, tf_map_real.dfbin, tf_map_real.dfbin*tf_map_real.nfbins]
    #print(extents)

    ### PyWavelet

    EMRI_timeseries = TimeSeries(signal.real, time)
    
    s = int(select_scale_df(target_df, fs)*res_factor)
    nfsize = 2**s
    ntsize = int(len(EMRI_timeseries) / nfsize)
    ntsize, length = _get_nzeros_padding(ntsize, 2**s, len(EMRI_timeseries))

    nz = length - len(EMRI_timeseries)
    if nz != 0:
        #print(f"Adding {nz} zeros at end")
        zeros=np.zeros(nz)
        newdata = np.concatenate((EMRI_timeseries, zeros))

    padded_timeseries = TimeSeries(newdata, time)

    EMRI_wavelet = np.array(from_time_to_wavelet(padded_timeseries, Nt= ntsize, Nf= nfsize))
    #print(EMRI_wavelet.transpose().shape)



    diff = np.abs(np.array(tf_map_real.data) - EMRI_wavelet.transpose())
    #print(diff.shape)
    

np.savez_compressed('./tfm_1y.npz', ltft=tf_map_real.data.transpose(), pywavelet=EMRI_wavelet, diff=diff.transpose())