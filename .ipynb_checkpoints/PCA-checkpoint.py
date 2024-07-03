from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def overlap(x, y):
    overlaps = []
    for i in range(len(x)):
        overlaps.append(np.dot(x[i], y[i])/(np.linalg.norm(x[i])*np.linalg.norm(y[i])))
    return overlaps



filename = '5_samples_td'
train_size = 3


matr = np.load('./H_matrices_td/' + filename + '.npz')

 

trunc_train = matr['H_trunc'][:train_size]
zeros_train = matr['H_zeros'][:train_size]


trunc_test = matr['H_trunc'][train_size:]
zeros_test = matr['H_zeros'][train_size:]

"""

zeros_test = matr['wavelets'][train_size:]
zeros_train = matr['wavelets'][:train_size]

"""


ol_measure=[]

n_components = np.arange(0, 3, 1)


# pca = PCA(n_components=n_components)

for i in range(len(n_components)):
    pca = PCA(n_components=n_components[i])
    fit = pca.fit(zeros_train)

    c_vec = pca.transform(zeros_test)

    h_hat = pca.inverse_transform(c_vec)

    ol_measure.append(np.sum(overlap(zeros_test, h_hat))/len(h_hat))




plt.plot(n_components, ol_measure)
plt.xlabel('no. of components kept')
plt.ylabel('normalized overlap of ' + str(len(h_hat)) + ' waveforms')
plt.hlines(0.97, xmin=0, xmax=n_components[-1], colors='tab:red')
plt.title(filename)
plt.savefig('./plots/' + filename + 'PCA')
