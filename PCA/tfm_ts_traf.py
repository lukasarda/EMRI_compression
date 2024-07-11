import numpy as np
import os
from helpers import read_global_extremes, inv_wavelet_transform

def main(tfm_path, ts_out_path):

    input_folder_path = '/sps/lisaf/lkarda/H_matrices_td/100_samples_1_months/global_extremes.txt'
    global_min, global_max= read_global_extremes(file_path= input_folder_path)

    tfm= np.load(tfm_path)
    tfm_array_in= tfm['input'][0]
    tfm_array_out= tfm['output'][0]

    print(tfm_array_in.shape)
    print(tfm_array_out.shape)


    ts_in= inv_wavelet_transform(
        tfm_array_in,
        global_min=global_min,
        global_max=global_max
    )

    ts_out= inv_wavelet_transform(
        tfm_array_out,
        global_min=global_min,
        global_max=global_max
    )

    tfm_filename = os.path.basename(tfm_path)
    ts_filename = 'ts_' + tfm_filename
    ts_output_path = os.path.join('/sps/lisaf/lkarda/test_bin/ts/', ts_filename)

    # Save the data
    np.savez_compressed(ts_output_path, output=ts_out.data, input=ts_in.data)



if __name__ == "__main__":
    # tfm_path= '/sps/lisaf/lkarda/test_bin/comp_in_out_PCA_1_components.npz'
    # ts_out_path= '/sps/lisaf/lkarda/test_bin/ts/1_ts.npz'

    # main(tfm_path, ts_out_path)


    path= '/sps/lisaf/lkarda/test_bin/'

    files_in_path= [f for f in os.listdir(path) if f.endswith('.npz')]

for n in files_in_path:

    main(path+ n, '/sps/lisaf/lkarda/test_bin/ts/ts_{}'.format(n[:-4]))

