import pygenten._pygenten as gt
from pygenten.utils import make_algparams

def make_guess(args):
    u = gt.Ktensor()
    rem = args
    if 'init' in rem:
        u = rem.pop('init')
    return u,rem

def make_dtc(args):
    dtc = None
    rem = args
    if 'dtc' in rem:
        dtc = rem.pop('dtc')
    return dtc,rem

def check_invalid(args):
    if len(args) > 0:
      msg = "Invalid solver arguments:  " + str(args)
      raise RuntimeError(msg)

def driver(X, u, args, dtc=None):
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
        u = gt.Ktensor(u.weights, u.factor_matrices, copy=False)
        is_ttb = True

    # Call Genten
    if dtc is not None:
        M,info = gt.driver(dtc, X, u, args);
    else:
        M,info = gt.driver(X, u, args);

    # Convert result Ktensor to TTB ktensor if necessary
    # We make a copy because the internal GenTen Ktensor will disappear
    if is_ttb:
        factor_matrices = []
        for i in range(0, M.ndims):
            factor_matrices.append(np.array(M[i]))
        M = ttb.ktensor.from_data(M.weights, factor_matrices, copy=True)

    return M, info

def cp_als(X, **kwargs):
    rem = kwargs
    u,rem = make_guess(rem)
    a,rem = make_algparams(rem)
    dtc,rem = make_dtc(rem)
    check_invalid(rem)

    a.method = gt.CP_ALS
    return driver(X, u, a, dtc)

def cp_opt(X, **kwargs):
    rem = kwargs
    u,rem = make_guess(rem)
    a,rem = make_algparams(rem)
    dtc,rem = make_dtc(rem)
    check_invalid(rem)

    a.method = gt.CP_OPT
    return driver(X, u, a, dtc)

def gcp_opt(X, **kwargs):
    rem = kwargs
    u,rem = make_guess(rem)
    a,rem = make_algparams(rem)
    dtc,rem = make_dtc(rem)
    check_invalid(rem)

    a.method = gt.GCP_OPT
    return driver(X, u, a, dtc)

def gcp_sgd(X, **kwargs):
    rem = kwargs
    u,rem = make_guess(rem)
    a,rem = make_algparams(rem)
    dtc,rem = make_dtc(rem)
    check_invalid(rem)

    a.method = gt.GCP_SGD
    return driver(X, u, a, dtc)
