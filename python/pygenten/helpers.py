import pyttb
import numpy

def make_ttb_tensor(X, copy=True):
    return pyttb.tensor.from_data(X.data, X.shape, copy=copy)

def make_ttb_sptensor(X, copy=True):
    if copy:
        return pyttb.sptensor.from_data(X.subs, X.vals, X.shape, copy=copy)
    else:
        return pyttb.sptensor.from_data(X.subs.copy(), X.vals.copy(), X.shape)

def make_ttb_ktensor(M, copy=True):
    return pyttb.ktensor.from_data(M.weights, M.factor_matrices, copy=copy)
