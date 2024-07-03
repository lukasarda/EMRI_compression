import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_colormap(tfm, extents, name):
    norm = LogNorm(vmin=np.min(np.abs(tfm)[np.abs(tfm) != 0]), vmax=np.max(np.abs(tfm)))
    plt.imshow(np.abs(tfm), aspect='auto', extent=extents, norm=norm, origin='lower')
    plt.yscale('log')
    #ax._colorbars()
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar()
    plt.savefig('/pbs/home/l/lkarda/EMRI_compression/plots/'+name+'.png')
    plt.close()


# file = np.load('/pbs/home/l/lkarda/EMRI_compression/AE_CNN_dc_1month_0.npz')

file = np.load('/pbs/home/l/lkarda/EMRI_compression/AE_CNN_maxPool_1month_0.npz')


extents = [0.0, 3164160.0, 9.765625e-05, 0.05]

# ltft = file['ltft']
# pywavelet = file['pywavelet']
# diff = file['diff']

# a = plot_colormap(ltft, extents, 'ltft')
# b = plot_colormap(pywavelet, extents, 'pywavelet')
# c = plot_colormap(diff, extents, 'diff')


AE = file['AE'][0][1]
print(AE.shape)
print(AE.mean())
d = plot_colormap(np.squeeze(AE), extents, 'test_decoded_maxpool_1')


