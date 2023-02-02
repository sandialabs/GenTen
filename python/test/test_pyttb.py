import pyttb as ttb
import numpy as np

weights = np.array([1., 2.])
fm0 = np.array([[1., 2.], [3., 4.]])
fm1 = np.array([[5., 6.], [7., 8.]])
K = ttb.ktensor.from_data(weights, [fm0, fm1])
np.random.seed(1)
print(K.full())
M, Minit, output = ttb.cp_als(K.full(), 2)