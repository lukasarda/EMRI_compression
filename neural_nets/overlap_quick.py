import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from dataset_class_nn import npz_class
from neural_net_class import AE_net
from num_features import cn_enc_out_shape
import importlib.util
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm



def load_model(module_path, model_class_name, model_state_name, channel_mult, num_fc_nodes_bottleneck, input_shape):
    h_cn_enc_out, w_cn_enc_out, h_kernel, w_kernel = cn_enc_out_shape(channel_mult=channel_mult, input_shape=input_shape)
    num_fc_nodes_after_conv = 100 * h_cn_enc_out * w_cn_enc_out

    spec = importlib.util.spec_from_file_location("neural_net_module", module_path)
    neural_net_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(neural_net_module)
    model_class = getattr(neural_net_module, model_class_name)
    model = model_class(
        channel_mult=channel_mult,
        h_kernel=h_kernel,
        w_kernel=w_kernel,
        num_fc_nodes_after_conv=num_fc_nodes_after_conv,
        num_fc_nodes_bottleneck=num_fc_nodes_bottleneck
    )
    
    model_state_path = os.path.join('/sps/lisaf/lkarda/model_states/res_states', model_state_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_state_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def mat_overlap(a, b):
    nom = np.multiply(a, b).sum()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return nom / denom

def plot_colormap(tfm, extents, ax, title):
    norm = LogNorm(vmin=np.min(np.abs(tfm)[np.abs(tfm) != 0]), vmax=np.max(np.abs(tfm)))
    cax = ax.imshow(np.abs(tfm), aspect='auto', extent=extents, norm=norm, origin='lower')
    ax.set_yscale('log')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    plt.colorbar(cax, ax=ax)

def main(module_path, model_class_name, model_state_name, input_folder, plot_flag=None):
    dir_path = os.path.join('/sps/lisaf/lkarda/H_matrices_td', input_folder, 'tfm_singles')
    dataset = npz_class(dirpath=dir_path)

    batch_size = 100
    num_workers = 4
    channel_mult = 8

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Assuming the first batch is used for model input and output
    input_tensor_tfm = next(iter(loader)).to(device)

    model = load_model(
        module_path=module_path,
        model_class_name=model_class_name,
        model_state_name=model_state_name,
        channel_mult=channel_mult,
        num_fc_nodes_bottleneck=5000,
        input_shape=input_tensor_tfm.shape[2:]
    )

    # Get model output
    with torch.no_grad():
        output_tensor_tfm = model(input_tensor_tfm.float())

    input_data_np = input_tensor_tfm.cpu().numpy()  # Move to CPU and remove added dimension
    output_data_np = output_tensor_tfm.cpu().numpy()  # Move to CPU and remove added dimension

    # real_overlap=[mat_overlap(input_data_np[i][0], output_data_np[i][0]) for i in range(input_data_np.shape[0])]
    # imag_overlap=[mat_overlap(input_data_np[i][1], output_data_np[i][1]) for i in range(input_data_np.shape[0])]


    # print(f"Real Overlap: {np.mean(np.array(real_overlap))}")
    # print(f"Imaginary Overlap: {np.mean(np.array(imag_overlap))}")


    id_sample = 10
    if plot_flag == 'tfm':
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        extents = [0.0, 3164160.0, 9.765625e-05, 0.05]  # Modify as needed


        plot_colormap(input_data_np[id_sample][0], extents, axs[0, 0], 'Real Input')
        plot_colormap(input_data_np[id_sample][1], extents, axs[0, 1], 'Imaginary Input')
        plot_colormap(output_data_np[id_sample][0], extents, axs[1, 0], 'Real Output')
        plot_colormap(output_data_np[id_sample][1], extents, axs[1, 1], 'Imaginary Output')
        
        real_overlap=mat_overlap(input_data_np[id_sample][0], output_data_np[id_sample][0])
        imag_overlap=mat_overlap(input_data_np[id_sample][1], output_data_np[id_sample][1])

        plt.tight_layout()
        fig.suptitle('Overlaps: Real: {}, Imaginary: {}'.format(real_overlap, imag_overlap))
        fig.tight_layout(rect=[0, 0.03, 1, 0.99])
        plt.savefig('/pbs/home/l/lkarda/EMRI_compression/plots/tfm_1_months_83epochs_5000.png')
        plt.close()

if __name__ == "__main__":
    module_path = '/pbs/home/l/lkarda/EMRI_compression/neural_nets/neural_net_class.py'  
    model_class_name = 'AE_net'
    model_state_name = 'AE_net_30000_samples_1_months_8_5000_20240630_133536_83'
    input_folder = '100_samples_1_months/'
    plot_flag = 'tfm'

    main(module_path, model_class_name, model_state_name, input_folder, plot_flag)
