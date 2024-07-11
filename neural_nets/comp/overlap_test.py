import numpy as np
from helpers_comp import mat_overlap


a = np.random.uniform(low=-1., high=1.,size=(512, 516))

b = np.random.uniform(low=-1., high=1.,size=(512, 516))
c= np.zeros((512,516))

print(mat_overlap(a,b))