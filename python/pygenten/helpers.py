import pyttb
import numpy

def make_ttb_tensor(X):
    return pyttb.tensor.from_data(numpy.array(X.values()), tuple(numpy.array(X.size())))

def make_ttb_ktensor(M, copy=True):
    return pyttb.ktensor.from_data(M.weights, M.factor_matrices, copy=copy)
