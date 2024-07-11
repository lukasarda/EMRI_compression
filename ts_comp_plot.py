import numpy as np
import matplotlib.pyplot as plt
import os

def plot_graph(ts, dt, ax, title):
    """
    Plot a time series on a given axis.

    Parameters:
    - ts: Time series data to plot.
    - dt: Time step between samples.
    - ax: Matplotlib axis to plot on.
    - title: Title for the plot.
    """
    times = np.linspace(0., len(ts), len(ts)) * dt
    ax.plot(times, ts)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)


def plot_graphs(file_path, name, dt=10.):
    """
    Plot time series for input and output arrays from a .npz file.

    Parameters:
    - file_path: Path to the .npz file.
    - dt: Time step between samples.
    - name: Name for saving the combined plot.
    - sample_idx: Index of the sample to plot.
    """

    file = np.load(file_path)
    output_real = file[list(file.keys())[0]][0]
    output_imag = file[list(file.keys())[0]][1]
    input_real = file[list(file.keys())[1]][0]
    input_imag = file[list(file.keys())[1]][1]

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    plot_graph(np.squeeze(input_real)[:1000], dt, ax=axs[0, 0], title='Input Real')
    plot_graph(np.squeeze(input_imag)[:1000], dt, ax=axs[1, 0], title='Input Imaginary')
    plot_graph(np.squeeze(output_real)[:1000], dt, ax=axs[0, 1], title='Output Real')
    plot_graph(np.squeeze(output_imag)[:1000], dt, ax=axs[1, 1], title='Output Imaginary')


    axs[0, 0].set_title('Input Real')
    axs[1, 0].set_title('Input Imaginary')
    axs[0, 1].set_title('Output Real')
    axs[1, 1].set_title('Output Imaginary')

    plt.tight_layout()
    plt.savefig('/pbs/home/l/lkarda/EMRI_compression/plots/' + name + '.png')
    plt.close()

# Example usage

# path= '/sps/lisaf/lkarda/H_matrices_td/30000_samples_1_months/singles/'

# files_in_path= [f for f in os.listdir(path) if f.endswith('.npz')]

# for n in files_in_path:
#     plot_graphs('/sps/lisaf/lkarda/test_bin/ts/' + n, 'in_out_PCA/ts_{}'.format(n[15:-4]))
#     print(n)
#     # print('/sps/lisaf/lkarda/test_bin/ts/' + n, 'in_out_PCA/ts_{}'.format(n[15:-4]))

a = np.load('/sps/lisaf/lkarda/H_matrices_td/30000_samples_1_months/singles/7000.npz')

b = np.load('/sps/lisaf/lkarda/H_matrices_td/30000_samples_1_months/singles/27998.npz')


fig, axs = plt.subplots(2, 2, figsize=(15, 12))

print(np.squeeze(a['ht'].real).shape)

plot_graph(np.squeeze(a['ht'].real)[:1000], 10, axs[0, 0], '7000')
plot_graph(np.squeeze(b['ht'].real)[:1000], 10, axs[0, 1], '27998')
plt.savefig('/pbs/home/l/lkarda/EMRI_compression/plots/' + 'two_waveforms' + '.png')


