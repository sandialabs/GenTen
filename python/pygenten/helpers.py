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
    return pyttb.tensor(X.data, X.shape, copy=copy)

def make_ttb_sptensor(X, copy=True):
    """
    Convert the given tensor to a pyttb.sptensor

    Parameters:
      * X: sparse tensor as a pygenten.Sptensor.
      * copy: whether the resulting tensor should be a copy of X or an alias.

    Returns a pyttb.sptensor for X.
    """
    return pyttb.sptensor(X.subs, X.vals, X.shape, copy=copy)

def make_ttb_ktensor(M, copy=True):
    """
    Convert the given ktensor to a pyttb.ktensor

    Parameters:
      * M: ktensor as a pygenten.Ktensor.
      * copy: whether the resulting ktensor should be a copy of M or an alias.

    Returns a pyttb.ktensor for M.
    """
    return  pyttb.ktensor(list(M.factor_matrices), M.weights, copy=copy)

def make_gt_tensor(X, copy=True):
    """
    Convert the given tensor to a pygenten.Tensor

    Parameters:
      * X: dense tensor as a pyttb.tensor.
      * copy: whether the resulting tensor should be a copy of X or an alias.

    Returns a pygenten.Tensor for X.
    """
    Xd = X.data
    if not Xd.flags.f_contiguous and not Xd.flags.c_contiguous:
        print('Warning:  dense tensor passed to make_gt_tensor() is not C or F-contiguous, so will be copied and reordered')
        Xd = numpy.asfortranarray(Xd)
    return gt.Tensor(Xd, copy=copy)

def make_gt_sptensor(X, copy=True):
    """
    Convert the given tensor to a pygenten.Sptensor

    Parameters:
      * X: sparse tensor as a pyttb.sptensor.
      * copy: whether the resulting tensor should be a copy of X or an alias.

    Returns a pygenten.Sptensor for X.
    """
    Xs = X.subs
    if not Xs.flags.c_contiguous:
        print('Warning:  sparse tensor subscripts passed to make_gt_sptensor() is not C-contiguous, so will be copied and reordered')
        Xs = numpy.ascontiguousnarray(Xs)
    return gt.Sptensor(X.shape, Xs, X.vals, copy=copy)

def make_gt_ktensor(M, copy=True):
    import copy as cpy
    """
    Convert the given ktensor to a pygenten.Ktensor

    Parameters:
      * M: ktensor as a pyttb.ktensor.
      * copy: whether the resulting ktensor should be a copy of M or an alias.

    Returns a pygenten.Ktensor for M.
    """
    Mf = cpy.copy(M.factor_matrices)
    for i in range(len(Mf)):
        if not Mf[i].flags.c_contiguous:
            print('Warning:  factor matrix passed to make_gt_ktensor() is not C-contiguous, so will be copied and reordered')
            Mf[i] = numpy.ascontiguousarray(Mf[i])
    return gt.Ktensor(M.weights, M.factor_matrices, copy=copy)
