"""
Helper functions for converting between pygenten and pyttb data structures.
"""

import pyttb
import numpy

def make_ttb_tensor(X, copy=True):
    """
    Convert the given tensor to a pyttb.tensor

    Parameters:
      * X: dense tensor as a pygenten.Tensor.
      * copy: whether the resulting tensor should be a copy of X or an alias.

    Returns a pyttb.tensor for X.
    """
    return pyttb.tensor.from_data(X.data, X.shape, copy=copy)

def make_ttb_sptensor(X, copy=True):
    """
    Convert the given tensor to a pyttb.sptensor

    Parameters:
      * X: sparse tensor as a pygenten.Sptensor.
      * copy: whether the resulting tensor should be a copy of X or an alias.

    Returns a pyttb.sptensor for X.
    """
    if copy:
        return pyttb.sptensor.from_data(X.subs, X.vals, X.shape, copy=copy)
    else:
        return pyttb.sptensor.from_data(X.subs.copy(), X.vals.copy(), X.shape)

def make_ttb_ktensor(M, copy=True):
    """
    Convert the given ktensor to a pyttb.ktensor

    Parameters:
      * M: ktensor as a pygenten.Ktensor.
      * copy: whether the resulting ktensor should be a copy of X or an alias.

    Returns a pyttb.ktensor for X.
    """
    return pyttb.ktensor.from_data(M.weights, M.factor_matrices, copy=copy)
