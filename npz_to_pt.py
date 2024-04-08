from helpers import npz_directory_to_tensor

data_dir = '/sps/lisaf/lkarda/H_matrices_td/5020_samples'
data_key = 'ht' # ht for time domain and hf for frequency domain

npz_directory_to_tensor(data_dir, data_key)
