import pyttb
import numpy

def make_ttb_tensor(X):
    return pyttb.tensor.from_data(numpy.array(X.values()), tuple(numpy.array(X.size())))

def make_ttb_ktensor(M, copy=True):
    nd = M.ndims()
    factors = []
    for i in range(nd):
        factors.append(numpy.array(M[i]))
    return pyttb.ktensor.from_data(numpy.array(M.weights()), factors, copy=copy)
