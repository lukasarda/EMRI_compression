import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

def plot_colormap(tfm, extents, name, ax=None):
    """
    Plot the colormap for a given array.

    Parameters:
    - tfm: 2D array to plot.
    - extents: List of extents for the axes.
    - name: Name for saving the plot (only used if ax is None).
    - ax: Matplotlib axis to plot on (if provided).
    """
    norm = LogNorm(vmin=np.min(np.abs(tfm)[np.abs(tfm) != 0]), vmax=np.max(np.abs(tfm)))
    if ax is None:
        plt.imshow(np.abs(tfm), aspect='auto', extent=extents, norm=norm, origin='lower')
        plt.yscale('log')
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar()
        plt.savefig('/pbs/home/l/lkarda/EMRI_compression/plots/' + name + '.png')
        plt.close()
    else:
        im = ax.imshow(np.abs(tfm), aspect='auto', extent=extents, norm=norm, origin='lower')
        ax.set_yscale('log')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        plt.colorbar(im, ax=ax)

def plot_colormaps(file_path, extents, name, sample_idx= 0):
    """
    Plot colormaps for input and output arrays from a .npz file.

    Parameters:
    - file_path: Path to the .npz file.
    - extents: List of extents for the axes.
    - name: Name for saving the combined plot.
    """

    file = np.load(file_path)
    output_real = file[list(file.keys())[0]][sample_idx][0]
    output_imag = file[list(file.keys())[0]][sample_idx][1]
    input_real = file[list(file.keys())[1]][sample_idx][0]
    input_imag = file[list(file.keys())[1]][sample_idx][1]

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    plot_colormap(np.squeeze(input_real), extents, 'input_real', ax=axs[0, 0])
    plot_colormap(np.squeeze(input_imag), extents, 'input_imag', ax=axs[1, 0])
    plot_colormap(np.squeeze(output_real), extents, 'output_real', ax=axs[0, 1])
    plot_colormap(np.squeeze(output_imag), extents, 'output_imag', ax=axs[1, 1])

    axs[0, 0].set_title('Input Real')
    axs[1, 0].set_title('Input Imaginary')
    axs[0, 1].set_title('Output Real')
    axs[1, 1].set_title('Output Imaginary')

    plt.tight_layout()
    plt.savefig('/pbs/home/l/lkarda/EMRI_compression/plots/' + name + '.png')
    plt.close()

# Example usage
extents = [0.0, 3164160.0, 9.765625e-05, 0.05]

path= '/sps/lisaf/lkarda/test_bin/'

files_in_path= [f for f in os.listdir(path) if f.endswith('.npz')]

for n in files_in_path:
    plot_colormaps('/sps/lisaf/lkarda/test_bin/' + n, extents, 'in_out_PCA/{}_components'.format(n[15:-4]), sample_idx=0)
    print(n)
