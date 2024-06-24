import numpy as np
import tempfile
import os

from wavelet_trafo_helpers import normalization, calculate_global_extremes, write_global_extremes


def main():
    dt = 10.  # s
    fs = float(1./10.)
    window = 'meyer'
    res_factor = 1.
    dir_name = '30000_samples_1_months'
    data_dir = '/sps/lisaf/lkarda/H_matrices_td/' + dir_name
    key = 'ht'
    file_list = [os.path.join(data_dir, 'singles', filename) for filename in os.listdir(os.path.join(data_dir, 'singles')) if os.path.isfile(os.path.join(data_dir, 'singles', filename))]
    tfm_dir = os.path.join(data_dir, 'tfm_singles')
    os.makedirs(tfm_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        global_max, global_min = calculate_global_extremes(temp_dir, file_list, dt, fs, window, res_factor, key)
        
        # Write global extremes to a text file
        write_global_extremes(data_dir, global_max, global_min)
        
        for i, file in enumerate(file_list):
            sample_id = os.path.splitext(os.path.basename(file))[0]
            temp_file = os.path.join(temp_dir, sample_id + '_tfm.npz')
            data = np.load(temp_file)
            
            tfm_real_normed = normalization(data['real'].transpose(), arr_min=global_min, arr_max=global_max, up=1, low=0)
            tfm_imag_normed = normalization(data['imag'].transpose(), arr_min=global_min, arr_max=global_max, up=1, low=0)
            np.savez_compressed(os.path.join(tfm_dir, sample_id + '_tfm.npz'), real=tfm_real_normed, imag=tfm_imag_normed)
            
            print(f'{i+1}/{len(file_list)}')

if __name__ == "__main__":
    main()