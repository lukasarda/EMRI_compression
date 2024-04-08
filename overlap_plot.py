import numpy as np
import matplotlib.pyplot as plt

file=str(5000)
overlaps = np.load('/sps/lisaf/lkarda/overlaps/'+file+'.npz')['overlaps']


print(overlaps)
plt.plot(overlaps)
plt.title(file)
plt.ylabel('Overlap')
plt.xlabel('# of processed batches')
plt.savefig('./plots/overlap_'+file+'_samples.png')
