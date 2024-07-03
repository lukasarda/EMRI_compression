import os
import numpy as np
from few.waveform import FastSchwarzschildEccentricFlux

from parameter import para_con

def save_waveforms(params, few, output_dir, save_freq):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(0, len(params), save_freq):
        chunk_params = params[i:i+save_freq]
        chunk_ht_sto = []

        for param in chunk_params:
            ht = few(param[0], param[1], param[2], param[3], param[4], param[5], dt=param[6], T=param[7])
            chunk_ht_sto.append(ht.get())

        np.savez_compressed(os.path.join(output_dir, f'{i+save_freq}.npz'), ht=chunk_ht_sto)
        print(f'Saved {i+save_freq}/{len(params)}')

def main(no_of_waveforms, save_freq):
    use_gpu = True

    inspiral_kwargs = {
        "DENSE_STEPPING": 0,
        "max_init_len": int(1e3),
    }

    amplitude_kwargs = {
        "max_init_len": int(1e3),
        "use_gpu": use_gpu,
    }

    Ylm_kwargs = {
        "assume_positive_m": False,
    }

    sum_kwargs = {
        "use_gpu": use_gpu,
        "pad_output": False,
    }

    few = FastSchwarzschildEccentricFlux(
        inspiral_kwargs=inspiral_kwargs,
        amplitude_kwargs=amplitude_kwargs,
        Ylm_kwargs=Ylm_kwargs,
        sum_kwargs=sum_kwargs,
        use_gpu=use_gpu,
    )



    params, obsT = para_con(no_of_waveforms)
    
    output_dir = '/sps/lisaf/lkarda/H_matrices_td/{}_samples_{}_months/singles'.format(no_of_waveforms, int(obsT*12.))

    save_waveforms(params, few, output_dir, save_freq)

if __name__ == "__main__":

    no_of_waveforms = 100
    save_freq = 1

    main(no_of_waveforms, save_freq)