import numpy as np
from parameter import para_con
import os


from few.waveform import FastSchwarzschildEccentricFlux, GenerateEMRIWaveform

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

few_gen = GenerateEMRIWaveform(
    "FastSchwarzschildEccentricFlux",
    sum_kwargs=dict(pad_output=True, output_type="fd", odd_len=True),
    return_list=True
)


few = FastSchwarzschildEccentricFlux(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu,
)


no_of_waveforms = 100

save_freq = 100


params = para_con(int(no_of_waveforms))

output_dir = '/sps/lisaf/lkarda/H_matrices_fd/' + str(len(params)) + '_samples'
os.mkdir(output_dir)

hf_sto = []

for i in range(0, len(params), save_freq):
    chunk_params = params[i:i+save_freq]
    chunk_hf_sto = []
    
    for param in chunk_params:
        a = 0.1  # will be ignored in Schwarzschild waveform
        x0 = 1.0  # will be ignored in Schwarzschild waveform
        qK = np.pi/3  # polar spin angle
        phiK = np.pi/3  # azimuthal viewing angle
        qS = np.pi/3  # polar sky angle
        phiS = np.pi/3  # azimuthal viewing angle
        dist = 1.0  # distance
        # initial phases
        Phi_phi0 = np.pi/3
        Phi_theta0 = 0.0
        Phi_r0 = np.pi/3

        waveform_kwargs = {
            "T": param[7],
            "dt": param[6],
        }

        emri_injection_params = [
            param[0],
            param[1],
            a,
            param[2],
            param[3],
            x0,
            dist,
            qS,
            phiS,
            qK,
            phiK,
            Phi_phi0,
            Phi_theta0,
            Phi_r0
        ]

        fd_kwargs = waveform_kwargs.copy()
        fd_kwargs['mask_positive'] = True

        hf = few_gen(*emri_injection_params, **fd_kwargs)
        chunk_hf_sto.append(list(hf))

    np.savez_compressed(os.path.join(output_dir, f'{i+save_freq}'), hf=chunk_hf_sto)
    print(f'Saved {i+save_freq}/{len(params)}')
        
