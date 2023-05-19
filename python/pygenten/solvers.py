"""
Solver functions for computing CP decompositions using GenTen.
"""

import pygenten._pygenten as gt
from pygenten.utils import make_algparams

def make_guess(args):
    """
    Utility function for extracting the initial guess from the given dict of
    keyword arguments.
    """
    u = gt.Ktensor()
    rem = args
    if 'init' in rem:
        u = rem.pop('init')
    return u,rem

def make_dtc(args):
    """
    Utility function for extracting the distributed tensor context from the
    given dict of keyword arguments.
    """
    dtc = None
    rem = args
    if 'dtc' in rem:
        dtc = rem.pop('dtc')
    return dtc,rem

def check_invalid(args):
    """
    Utility function for determing whether arguments were valid.
    """
    if len(args) > 0:
      msg = "Invalid solver arguments:  " + str(args)
      raise RuntimeError(msg)

def driver(X, u, args, dtc=None):
    """
    Wrapper function for calling GenTen's CP solver driver.

    This should not be called directly.  Call individual solver methods
    corresponding to the chosen algorithm instead.
    """
    try:
        import pyttb as ttb
        import numpy as np
        have_ttb = True
    except:
        have_ttb = False

    # Convert TTB tensor/sptensor to Genten Tensor/Sptensor
    is_ttb = False
    if have_ttb and isinstance(X, ttb.tensor):
        X = gt.Tensor(X.data)
        is_ttb = True
    if have_ttb and isinstance(X, ttb.sptensor):
        X = gt.Sptensor(X.shape, X.subs, X.vals)
        is_ttb = True

    # Convert TTB ktensor to Genten Ktensor
    if have_ttb and isinstance(u, ttb.ktensor):
        # This needs to be a copy for some reason I do not fully understand
        u = gt.Ktensor(u.weights, u.factor_matrices, copy=True)
        is_ttb = True

    # Call Genten
    if dtc is not None:
        M,info = gt.driver(dtc, X, u, args);
    else:
        M,info = gt.driver(X, u, args);

    # Convert result Ktensor to TTB ktensor if necessary
    # Copying shouldn't be necessary since the python objects carry
    # reference counts for the original GenTen objects, but it appears
    # to not always work.
    if is_ttb:
        M = ttb.ktensor.from_data(M.weights, M.factor_matrices, copy=True)

    return M, info

def cp_als(X, **kwargs):
    """
    Compute CP decomposition using Alternating Least-Squares algorithm.

    Parameters:
      * X: Tensor to compute decomposition from.  May be provided using
        pyttb or pygenten tensor classes (pyttb.tensor/pygenten.Tensor and
        pyttb.sptensor/pygenten.Sptensor).
      * kwargs:  Keyword arguments for optional arguments including:
        * initial guess (as a pyttb.ktensor or pygenten.Ktensor).
        * distributed tensor context.
        * algorithmic parameters as pygenten.AlgParams.
        * any named parameters stored within pygenten.AlgParams.

    Returns:
      * u:  the ktensor solution (as either a pyttb.ktensor
        or pygenten.ktensor, depending on what was passed in).
      * perfInfo:  performance information as pygenten.PerfHistory.
    """
    rem = kwargs
    u,rem = make_guess(rem)
    a,rem = make_algparams(rem)
    dtc,rem = make_dtc(rem)
    check_invalid(rem)

    a.method = gt.CP_ALS
    return driver(X, u, a, dtc)

def cp_opt(X, **kwargs):
    """
    Compute CP decomposition using all-at-once gradient-based optimization.

    Parameters:
      * X: Tensor to compute decomposition from.  May be provided using
        pyttb or pygenten tensor classes (pyttb.tensor/pygenten.Tensor and
        pyttb.sptensor/pygenten.Sptensor).
      * kwargs:  Keyword arguments for optional arguments including:
        * initial guess (as a pyttb.ktensor or pygenten.Ktensor).
        * distributed tensor context.
        * algorithmic parameters as pygenten.AlgParams.
        * any named parameters stored within pygenten.AlgParams.

    Returns:
      * u:  the ktensor solution (as either a pyttb.ktensor
        or pygenten.ktensor, depending on what was passed in).
      * perfInfo:  performance information as pygenten.PerfHistory.
    """
    rem = kwargs
    u,rem = make_guess(rem)
    a,rem = make_algparams(rem)
    dtc,rem = make_dtc(rem)
    check_invalid(rem)

    a.method = gt.CP_OPT
    return driver(X, u, a, dtc)

def gcp_opt(X, **kwargs):
    """
    Compute Generalized CP decomposition using all-at-once gradient-based
    optimization.

    Parameters:
      * X: Tensor to compute decomposition from.  May be provided using
        pyttb or pygenten tensor classes (pyttb.tensor/pygenten.Tensor and
        pyttb.sptensor/pygenten.Sptensor).
      * kwargs:  Keyword arguments for optional arguments including:
        * initial guess (as a pyttb.ktensor or pygenten.Ktensor).
        * distributed tensor context.
        * algorithmic parameters as pygenten.AlgParams.
        * any named parameters stored within pygenten.AlgParams.

    Returns:
      * u:  the ktensor solution (as either a pyttb.ktensor
        or pygenten.ktensor, depending on what was passed in).
      * perfInfo:  performance information as pygenten.PerfHistory.
    """
    rem = kwargs
    u,rem = make_guess(rem)
    a,rem = make_algparams(rem)
    dtc,rem = make_dtc(rem)
    check_invalid(rem)

    a.method = gt.GCP_OPT
    return driver(X, u, a, dtc)

def gcp_sgd(X, **kwargs):
    """
    Compute Generalized CP decomposition using all-at-once stochastic gradient
    descent-based optimization.

    Parameters:
      * X: Tensor to compute decomposition from.  May be provided using
        pyttb or pygenten tensor classes (pyttb.tensor/pygenten.Tensor and
        pyttb.sptensor/pygenten.Sptensor).
      * kwargs:  Keyword arguments for optional arguments including:
        * initial guess (as a pyttb.ktensor or pygenten.Ktensor).
        * distributed tensor context.
        * algorithmic parameters as pygenten.AlgParams.
        * any named parameters stored within pygenten.AlgParams.

    Returns:
      * u:  the ktensor solution (as either a pyttb.ktensor
        or pygenten.ktensor, depending on what was passed in).
      * perfInfo:  performance information as pygenten.PerfHistory.
    """
    rem = kwargs
    u,rem = make_guess(rem)
    a,rem = make_algparams(rem)
    dtc,rem = make_dtc(rem)
    check_invalid(rem)

    a.method = gt.GCP_SGD
    return driver(X, u, a, dtc)
