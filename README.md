# EMRI_compression
Various Compression and Analysis Codes for EMRIs

Scripts for waveform generation, preprocessing of the data and analysis with the Incremental PCA method.

DISCLAIMER:
Many directory locations are hard-coded and need to be changed accordingly.


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
