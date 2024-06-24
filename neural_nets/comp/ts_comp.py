import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from ltft.timeserie import Timeserie
from ltft.tfmap import TimeFrequencyArray
from helpers_comp import load_model, plot_colormap, mat_overlap, plot_graph
from wavelet_trafo import wavelet_transform, inv_process_signal, process_signal, inverse_normalization, read_global_extremes, load_file, inv_wavelet_transform
from ltft.common.utils import select_scale_df


def main(model_class_name, model_state_name, input_file, output_plot, plot_flag):
    # Load model
    module_path = '/pbs/home/l/lkarda/EMRI_compression/neural_nets/neural_net_class.py'
    input_file_path = '/sps/lisaf/lkarda/H_matrices_td/' + input_file
    input_folder_path = '/sps/lisaf/lkarda/H_matrices_td/' + input_file.split(os.sep)[0] + '/global_extremes.txt'
    output_plot_path = '/pbs/home/l/lkarda/EMRI_compression/plots/' + output_plot
    
    global_min, global_max= read_global_extremes(file_path= input_folder_path)


    sample_id, input_signal= load_file(input_file_path)

    tfms, cfgs = wavelet_transform(input_signal, global_min=global_min, global_max=global_max)
    input_tensor_tfm = torch.tensor(tfms, dtype=torch.float32).unsqueeze(0)

    model = load_model(module_path, model_class_name, model_state_name, channel_mult= 8, num_fc_nodes_bottleneck= 1000, input_shape= tfms[0].shape)
    # Get model output
    with torch.no_grad():
        output_tensor_tfm = model(input_tensor_tfm)
    
    # Convert tensors to numpy arrays
    input_data_np = input_tensor_tfm.squeeze(0).numpy()  # Remove added dimension
    output_data_np = output_tensor_tfm.squeeze(0).numpy()  # Remove added dimension
    
    output_signal = inv_wavelet_transform(output_data_np, global_min, global_max, cfgs=cfgs, dt=10., fs=1./10., window='meyer', res_factor=1.)


    # if plot_flag == 'tfm':
    #     # Plot input and output
    #     fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    #     extents = [0.0, 3164160.0, 9.765625e-05, 0.05]  # Modify as needed


    #     plot_colormap(input_data_np[0], extents, axs[0, 0], 'Real Input')
    #     plot_colormap(input_data_np[1], extents, axs[0, 1], 'Imaginary Input')
    #     plot_colormap(output_data_np[0], extents, axs[1, 0], 'Real Output')
    #     plot_colormap(output_data_np[1], extents, axs[1, 1], 'Imaginary Output')
        
    #     real_overlap=mat_overlap(input_data_np[0], output_data_np[0])
    #     imag_overlap=mat_overlap(input_data_np[1], output_data_np[1])

    #     plt.tight_layout()
    #     fig.suptitle('Overlaps: Real: {}, Imaginary: {}'.format(real_overlap, imag_overlap))
    #     fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    #     plt.savefig(output_plot_path)
    #     plt.close()

    # if plot_flag == 'ts':
    #     # Plot input and output
    #     fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    #     dt = 10. #s

    #     plot_graph(input_signal.real[0:1000], dt, axs[0, 0], 'Real Input')
    #     plot_graph(input_signal.imag[0:1000], dt, axs[0, 1], 'Imaginary Input')
    #     plot_graph(output_signal[0].data[0:1000], dt, axs[1, 0], 'Real Output')
    #     plot_graph(output_signal[1].data[0:1000], dt, axs[1, 1], 'Imaginary Output')
        
    #     real_overlap=mat_overlap(input_signal.real, output_signal[0].data)
    #     imag_overlap=mat_overlap(input_signal.imag, output_signal[1].data)

    #     plt.tight_layout()
    #     fig.suptitle('Overlaps: Real: {}, Imaginary: {}'.format(real_overlap, imag_overlap))
    #     fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    #     plt.savefig(output_plot_path)
    #     plt.close()



if __name__ == "__main__":
    
    model_class_name = 'AE_net'
    model_state_name = 'AE_CNN_maxPool2_20240617_132507_79'
    input_file = '5020_samples/singles/35_19_sample.npz'
    output_plot = 'tfm_bigdataset_80epochs.png'
    plot_flag = 'tfm' # 'tfm' or 'ts'

    main(model_class_name, model_state_name, input_file, output_plot, plot_flag)