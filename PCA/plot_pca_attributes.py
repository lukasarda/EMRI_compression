import numpy as np
import matplotlib.pyplot as plt

ipca_file= np.load('/sps/lisaf/lkarda/H_matrices_td/30000_samples_1_months/ipca_1000_components.npz')

singular_values= ipca_file['singular_values']
name= 'singular_values_1000'

plt.plot(range(len(singular_values)), singular_values)
plt.ylabel('Singular Values')
plt.xlabel('# of component')
plt.savefig('/pbs/home/l/lkarda/EMRI_compression/plots/' + name + '.png')