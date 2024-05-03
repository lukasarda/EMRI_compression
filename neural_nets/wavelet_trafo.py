import numpy as np
import os

from wavelet_trafo_helpers import wavelet_trafo, list_files

dt = 10. #s
fs = float(1./10.)
window = 'meyer'
res_factor = 1.



dir_name= '30000_samples'
data_dir= '/sps/lisaf/lkarda/H_matrices_td/'+dir_name
key= 'key'

file_list = list_files(data_dir+'/singles/')

tfm_dir= data_dir+'/tfm_singles/'
os.makedirs(tfm_dir, exist_ok=True)


for i, file in enumerate(file_list):
    sample_id= os.path.splitext(os.path.basename(file))[0]
    signal= np.load(file)[key]

    ### LTFT

    tf_map_real= wavelet_trafo(window, signal.real, dt, fs, res_factor=res_factor)
    tf_map_imag= wavelet_trafo(window, signal.imag, dt, fs, res_factor=res_factor)
    
    # print(tf_map_real.data.shape)
    # print(tf_map_imag.data.shape)

    print(i,'/',len(file_list))
    np.savez_compressed(tfm_dir+sample_id+'_tfm.npz', real=tf_map_real.data.transpose(), imag=tf_map_imag.data.transpose())