import pygenten
import numpy as np

tensor = pygenten.Ktensor(10,2)

weights = np.array(tensor.weights(), copy=False)
print(weights)
weights *= 10.
weights2 = np.array(tensor.weights(), copy=False)
print(weights2)
weights3 = np.zeros((10,))
tensor.setWeights(pygenten.Array(weights3))
weights4 = np.array(tensor.weights(), copy=False)
print(weights4)
del weights, weights2, weights4, tensor
