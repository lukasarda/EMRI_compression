import numpy as np
from pywavelet.transforms import from_time_to_wavelet
from pywavelet.transforms.types import TimeSeries

filename = '5_samples_td'

EMRI_waveform = np.load('./H_matrices_td/' + filename + '.npz')['H_trunc']
EMRI_time = np.load('./H_matrices_td/' + filename +  '.npz')['time']


wavelets = []

for a in range(len(EMRI_waveform)):

    EMRI_timeseries = TimeSeries(EMRI_waveform[a][:int(len(EMRI_waveform[a])/2)], time=EMRI_time)

    req_size = int(np.sqrt(len(EMRI_timeseries))+1) * int(np.sqrt(len(EMRI_timeseries)))

    gg = np.zeros(req_size)
    gg[:len(EMRI_timeseries)] = EMRI_timeseries

    gg_time = np.zeros(req_size)
    gg_time[:len(EMRI_time)] = EMRI_time

    gg_timeseries = TimeSeries(gg, time=gg_time)

    EMRI_wavelet = from_time_to_wavelet(gg_timeseries, Nt=int(np.sqrt(len(EMRI_time))))

    wavelets.append(np.array(EMRI_wavelet).flatten())

    print(str(a+1)+'/'+str(len(EMRI_waveform)))



np.savez_compressed('./H_matrices_wavelet/' + filename + '_wavelet', wavelets = np.array(wavelets))
