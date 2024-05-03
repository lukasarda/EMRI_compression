import numpy as np
import os

from ltft.timeserie import Timeserie
from ltft.common.utils import select_scale_df, get_df_bin, get_dt_bin



def wavelet_trafo(window, signal, dt, fs, res_factor=1.):

    ts = Timeserie(signal, dt=dt)

    target_df = 1.e-4 # Hz
    s = int(select_scale_df(target_df, fs)*res_factor)
    # print(f"Selected scale = {s}")

    DT = get_dt_bin(s, fs)
    DF = get_df_bin(s, fs)

    # print(f"Time resolution = {DT} s")
    # print(f"Frequency resolution = {DF} Hz")

    tfm = ts.dwilt(s, window, q=32)
    return tfm

dt = 10. #s
fs = float(1./10.)
window = 'meyer'


def list_files(directory):
    return [os.path.join(directory, filename) for filename in os.listdir(directory) if os.path.isfile(os.path.join(directory, filename))]
