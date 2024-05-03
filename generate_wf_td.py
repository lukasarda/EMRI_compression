import numpy as np
from parameter import para_con
import os

from few.waveform import FastSchwarzschildEccentricFlux

use_gpu = True

# keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
inspiral_kwargs={
        "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
        "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    }

# keyword arguments for inspiral generator (RomanAmplitude)
amplitude_kwargs = {
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    "use_gpu": use_gpu  # GPU is available in this class
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": False,
}


few = FastSchwarzschildEccentricFlux(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu,
)

no_of_waveforms = 3

save_freq = 1

params = para_con(int(no_of_waveforms))

output_dir = '/sps/lisaf/lkarda/H_matrices_td/' + str(len(params)) + '_samples'
os.mkdir(output_dir)


for i in range(0, len(params), save_freq):
    chunk_params = params[i:i+save_freq]
    chunk_ht_sto = []

    for param in chunk_params:
        ht = few(param[0], param[1], param[2], param[3], param[4], param[5], dt=param[6], T=param[7])
        chunk_ht_sto.append(list(ht.get()))

    np.savez_compressed(os.path.join(output_dir, f'{i+save_freq}'), ht=chunk_ht_sto)
    print(f'Saved {i+save_freq}/{len(params)}')