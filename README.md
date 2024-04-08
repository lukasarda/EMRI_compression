# EMRI_compression
Various Compression and Analysis Codes for EMRIs

Scripts for waveform generation, preprocessing of the data and analysis with the Incremental PCA method.

# Quick Guide
Quick guide how to perform the different analysis/compression methods.
---Generate Data---
1. Modify "parameter.py" to select your EMRI parameter space. Different distributions are available.
   In the current state only mass is varied inbetween parameter sets.
2. Generate data either in time domain "generate_wf_td.py" or frequency domain "generate_wf_fd.py".
   Inside the file four main variables should be specified. If GPUs should be used for generation "use_gpu",
   the number of samples to generate "no_of_waveforms" and the saving frequency i.e. the number of samples per file
   "save_freq".


---Incremental PCA---

1. 



DISCLAIMER:
Many directory locations are hard-coded and need to be changed accordingly.

# Installation manual
To create a conda environment on the CC-in2p3 cluster applicable to the codes follow this manual:

1. Go to an interactive gpu mode: srun -p gpu_interactive --gres=gpu:v100:1 -t 0-09:00 -n 4 --mem 40G --pty bash -i
2. Clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
3. cd FastEMRIWaveforms
4. Open install.sh and specify in line 102
    1. gcc_linux-64=12
    2. gxx_linux-64=12
    3. python=10
5. Add: export CUDAHOME=/usr/local/cuda-12.2/ to .bashrc in your home directory
6. Run source ~/.bashrc
7. Run bash install.sh env_name=few_pytorch
8. In /src/interpolate.cu add #include<stdexcept> to top of script
9. Run python setup.py install —ccbin ~/miniconda/envs/my_env/bin/x86_64-conda-linux-gnu-gcc
10. Run conda install nvidia::libcumlprims
11. Run conda install -c rapidsai cuml
12. Run conda install pytorch::pytorch
13. AGAIN Run python setup.py install —ccbin ~/miniconda/envs/my_env/bin/x86_64-conda-linux-gnu-gcc
14. Run python -m unittest discover
15. Done!
