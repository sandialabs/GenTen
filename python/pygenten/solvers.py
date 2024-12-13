"""
Solver functions for computing CP decompositions using GenTen.
"""

import pygenten._pygenten as gt
from pygenten.utils import make_algparams
try:
    from pygenten.helpers import make_ttb_ktensor, make_gt_tensor, make_gt_sptensor, make_gt_ktensor
except:
    pass

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
    
def convert_to_gt(arg):
    """
    Utility function to convert pyttb tensor/sptensor/ktensor to corresponding GenTen data structure
    """
    try:
        import pyttb as ttb
        import numpy as np
        have_ttb = True
    except:
        have_ttb = False

    if have_ttb:
        import pygenten.helpers

    is_ttb = False
    arg_gt = arg
    if have_ttb and isinstance(arg, ttb.tensor):
        arg_gt = make_gt_tensor(arg, copy=False)
        is_ttb = True
    if have_ttb and isinstance(arg, ttb.sptensor):
        arg_gt = make_gt_sptensor(arg, copy=False)
        is_ttb = True
    if have_ttb and isinstance(arg, ttb.ktensor):
        arg_gt = make_gt_ktensor(arg, copy=False)
        is_ttb = True
    return arg_gt,is_ttb

def driver(X, u, args, dtc=None):
    """
    Wrapper function for calling GenTen's CP solver driver.

    This should not be called directly.  Call individual solver methods
    corresponding to the chosen algorithm instead.
    """
    # Convert TTB data structures to GenTen (being careful to not overwrite arguments)
    X_gt,is_ttb_X = convert_to_gt(X)
    u_gt,is_ttb_u = convert_to_gt(u)

    # Call Genten
    if dtc is not None:
        M,M0,info = gt.driver(dtc, X_gt, u_gt, args)
    else:
        M,M0,info = gt.driver(X_gt, u_gt, args)

    # Convert result Ktensor to TTB ktensor if necessary
    if is_ttb_X or is_ttb_u:
        import pygenten.helpers
        M = make_ttb_ktensor(M, copy=False)
        M0 = make_ttb_ktensor(M0, copy=False)

    return M, M0, info

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
      * u0:  the initial guess (as either a pyttb.ktensor
        or pygenten.ktensor, depending on what was passed in).
      * perfInfo:  performance information as pygenten.PerfHistory.
    """
    rem = kwargs
    u,rem = make_guess(rem)
    dtc,rem = make_dtc(rem)
    a = make_algparams(rem)

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
      * u0:  the initial guess (as either a pyttb.ktensor
        or pygenten.ktensor, depending on what was passed in).
      * perfInfo:  performance information as pygenten.PerfHistory.
    """
    rem = kwargs
    u,rem = make_guess(rem)
    dtc,rem = make_dtc(rem)
    a = make_algparams(rem)

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
      * u0:  the initial guess (as either a pyttb.ktensor
        or pygenten.ktensor, depending on what was passed in).
      * perfInfo:  performance information as pygenten.PerfHistory.
    """
    rem = kwargs
    u,rem = make_guess(rem)
    dtc,rem = make_dtc(rem)
    a = make_algparams(rem)

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
      * u0:  the initial guess (as either a pyttb.ktensor
        or pygenten.ktensor, depending on what was passed in).
      * perfInfo:  performance information as pygenten.PerfHistory.
    """
    rem = kwargs
    u,rem = make_guess(rem)
    dtc,rem = make_dtc(rem)
    a = make_algparams(rem)

    a.method = gt.GCP_SGD
    return driver(X, u, a, dtc)

def gcp_fed(X, **kwargs):
    """
    Compute Generalized CP decomposition using all-at-once stochastic gradient
    descent-based optimization and federated learning.

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
      * u0:  the initial guess (as either a pyttb.ktensor
        or pygenten.ktensor, depending on what was passed in).
      * perfInfo:  performance information as pygenten.PerfHistory.
    """
    rem = kwargs
    u,rem = make_guess(rem)
    dtc,rem = make_dtc(rem)
    a = make_algparams(rem)

    a.method = gt.GCP_FED
    return driver(X, u, a, dtc)

def online_gcp(X, X0, u0, temporalAlgParams, spatialAlgParams, **kwargs):
    """
    Compute Generalized CP decomposition using streaming method.

    Parameters:
      * X: List of tensors to compute decomposition from.  May be provided using
        pyttb or pygenten tensor classes (pyttb.tensor/pygenten.Tensor and
        pyttb.sptensor/pygenten.Sptensor).
      * X0: Tensor used for warm start
      * u0: k-tensor warm start
      * temporalAlgParams: dict of algParams for temporal solver
      * spatialAlgParams: dict of algParams for spatial solver
      * kwargs:  algorithmic parameters for the streaming solver

    Returns:
      * u:  the ktensor solution (as either a pyttb.ktensor
        or pygenten.ktensor, depending on what was passed in).
      * estimated objective function at each step
      * tensor contribution to the objective
    """
    # make temporal/spatial AlgParams
    at = make_algparams(temporalAlgParams)
    ap = make_algparams(spatialAlgParams)

    # make streaming AlgParams
    rem = kwargs
    a = make_algparams(rem)

    # Convert pyttb to GenTen data structures (being careful to not overwrite arguments)
    X0_gt,is_ttb = convert_to_gt(X0)
    X_gt = [None]*len(X)
    for i in range(len(X)):
        X_gt[i],is_ttb_res = convert_to_gt(X[i])
        is_ttb = is_ttb or is_ttb_res
    u0_gt,is_ttb_res = convert_to_gt(u0)
    is_ttb = is_ttb or is_ttb_res

    # Call Genten
    u,fest,ften = gt.online_gcp_driver(X_gt,X0_gt,u0_gt,a,at,ap)

    # Convert result Ktensor to TTB ktensor if necessary
    if is_ttb:
        import pygenten.helpers
        u = make_ttb_ktensor(u, copy=False)
    
    return u,fest,ften

