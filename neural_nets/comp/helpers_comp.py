import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import importlib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from num_features import cn_enc_out_shape


def load_model(module_path, model_class_name, model_state_name, channel_mult, num_fc_nodes_bottleneck, input_shape):
    h_cn_enc_out, w_cn_enc_out, h_kernel, w_kernel = cn_enc_out_shape(channel_mult= channel_mult, input_shape= input_shape)
    num_fc_nodes_after_conv= 100 * h_cn_enc_out * w_cn_enc_out

    spec = importlib.util.spec_from_file_location("neural_net_module", module_path)
    neural_net_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(neural_net_module)
    model_class = getattr(neural_net_module, model_class_name)
    model = model_class(
        channel_mult= channel_mult,
        h_kernel= h_kernel,
        w_kernel= w_kernel,
        num_fc_nodes_after_conv= num_fc_nodes_after_conv,
        num_fc_nodes_bottleneck= num_fc_nodes_bottleneck
    )
    
    # Construct the full path to the model state
    model_state_path = '/sps/lisaf/lkarda/model_states/res_states/' + model_state_name
    
    model.load_state_dict(torch.load(model_state_path, map_location=torch.device('cuda')))
    model.eval()
    return model

def plot_colormap(tfm, extents, ax, title):
    norm = LogNorm(vmin=np.min(np.abs(tfm)[np.abs(tfm) != 0]), vmax=np.max(np.abs(tfm)))
    cax = ax.imshow(np.abs(tfm), aspect='auto', extent=extents, norm=norm, origin='lower')
    ax.set_yscale('log')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    plt.colorbar(cax, ax=ax)

def plot_graph(ts, dt, ax, title):
    times = np.linspace(0., len(ts), len(ts)) * dt
    cax = ax.plot(times, ts)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)


def mat_overlap(a,b):
    nom= np.multiply(a,b).sum()
    denom= np.linalg.norm(a) * np.linalg.norm(b)
    return nom/denom

def load_tfm(input_file_path):
    with np.load(input_file_path) as tfm:
        real = tfm[tfm.files[0]]
        imag = tfm[tfm.files[1]]
    combi = np.array([real, imag])

    # Prepare data for the model
    input_tensor = torch.from_numpy(combi).float().unsqueeze(0)
    return input_tensor
