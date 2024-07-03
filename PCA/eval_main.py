import numpy as np
import os
import cupy as cp
import matplotlib.pyplot as plt

def load_npz(path):
    file = np.load(path)
    return file[list(file.keys())[0]]

def mat_overlap(a,b):
    nom= np.multiply(a,b).sum()
    denom= np.linalg.norm(a) * np.linalg.norm(b)
    return nom/denom
    # a= cp.array(a)
    # b= cp.array(b)
    # nom= cp.multiply(a,b).sum()
    # denom= cp.linalg.norm(a) * cp.linalg.norm(b)
    # return nom/denom

def comp_overlap(data, traf):
    trafo= data@traf.T
    inv_trafo= trafo@traf
    return mat_overlap(data, inv_trafo)

def main():

    traf_matrix_path= '/sps/lisaf/lkarda/H_matrices_td/30000_samples_1_months/components_.npz'
    data_folder= '/sps/lisaf/lkarda/H_matrices_td/20_samples_1_months/tfm_singles'

    traf_matrix= load_npz(traf_matrix_path)

    data_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.npz')]
    data_stack = np.vstack([load_npz(file_path).flatten() for file_path in data_files])


    n_components= []
    overlaps= []
    for i in range(1, traf_matrix.shape[0]):
        n_components.append(traf_matrix.shape[0] - traf_matrix[:i].shape[0])
        overlaps.append(comp_overlap(data_stack, traf_matrix[:i]))

    plt.scatter(n_components, overlaps)
    plt.xlabel('# of components')
    plt.ylabel('Overlap')
    plt.xlim(0., traf_matrix.shape[0])
    plt.ylim(0., 1.1)
    # plt.hlines(1.)

    plt.savefig('/pbs/home/l/lkarda/EMRI_compression/plots/PCA_overlaps.png')
    






if __name__ == "__main__":
    main()