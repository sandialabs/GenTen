"""
Helper functions for converting between pygenten and pyttb data structures.
"""

import pygenten._pygenten as gt
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
        return pyttb.sptensor.from_data(X.subs.copy(), X.vals.copy(), X.shape)
    else:
        return pyttb.sptensor.from_data(X.subs, X.vals, X.shape)

def make_ttb_ktensor(M, copy=True):
    """
    Convert the given ktensor to a pyttb.ktensor

    Parameters:
      * M: ktensor as a pygenten.Ktensor.
      * copy: whether the resulting ktensor should be a copy of M or an alias.

    Returns a pyttb.ktensor for M.
    """
    return pyttb.ktensor.from_data(M.weights, M.factor_matrices, copy=copy)

def make_gt_tensor(X, copy=True):
    """
    Convert the given tensor to a pygenten.Tensor

    Parameters:
      * X: dense tensor as a pyttb.tensor.
      * copy: whether the resulting tensor should be a copy of X or an alias.

    Returns a pygenten.Tensor for X.
    """
    return gt.Tensor(X.data, copy=copy)

def make_gt_sptensor(X, copy=True):
    """
    Convert the given tensor to a pygenten.Sptensor

    Parameters:
      * X: sparse tensor as a pyttb.sptensor.
      * copy: whether the resulting tensor should be a copy of X or an alias.

    Returns a pygenten.Sptensor for X.
    """
    return gt.Sptensor(X.shape, X.subs, X.vals, copy=copy)

def make_gt_ktensor(M, copy=True):
    """
    Convert the given ktensor to a pygenten.Ktensor

    Parameters:
      * M: ktensor as a pyttb.ktensor.
      * copy: whether the resulting ktensor should be a copy of M or an alias.

    Returns a pygenten.Ktensor for M.
    """
    return gt.Ktensor(M.weights, M.factor_matrices, copy=copy)
