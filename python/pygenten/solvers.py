import pygenten._pygenten as gt

def make_algparams(args):
    a = gt.AlgParams()
    rem = {}
    for key,value in args.items():
        if hasattr(a, key):
            setattr(a, key, value)
        else:
            rem[key] = value
    return a, rem

def make_guess(args):
    u = gt.Ktensor()
    rem = args
    if 'init' in rem:
        u = rem.pop('init')
    return u,rem

def check_invalid(args):
    if len(args) > 0:
      msg = "Invalid solver arguments:  " + str(args)
      raise RuntimeError(msg)

def driver(X, u, args):
    try:
        import pyttb as ttb
        import numpy as np
        have_ttb = True
    except:
        have_ttb = False

    # Convert TTB tensor/sptensor to Genten Tensor/Sptensor
    if have_ttb and isinstance(X, ttb.tensor):
        X = gt.Tensor(X.data)
    if have_ttb and isinstance(X, ttb.sptensor):
        X = gt.Sptensor(X.shape, X.subs, X.vals)

    # Convert TTB ktensor to Genten Ktensor
    ttb_ktensor = False
    if have_ttb and isinstance(u, ttb.ktensor):
        u = gt.Ktensor(u.weights, u.factor_matrices)
        ttb_ktensor = True

    # Call Genten
    M,info = gt.driver(X, u, args);

    # Convert result Ktensor to TTB ktensor if necessary
    if ttb_ktensor:
        weights = np.array(M.weights(), copy=True)
        factor_matrices = []
        for i in range(0, M.ndims()):
            factor_matrices.append(np.array(M[i], copy=True))
        M = ttb.ktensor.from_data(weights, factor_matrices)

    return M, info

def cp_als(X, **kwargs):
    rem = kwargs
    u,rem = make_guess(rem)
    a,rem = make_algparams(rem)
    check_invalid(rem)

    a.method = gt.CP_ALS
    return driver(X, u, a)

def cp_opt(X, **kwargs):
    rem = kwargs
    u,rem = make_guess(rem)
    a,rem = make_algparams(rem)
    check_invalid(rem)

    a.method = gt.CP_OPT
    return driver(X, u, a)

def gcp_opt(X, **kwargs):
    rem = kwargs
    u,rem = make_guess(rem)
    a,rem = make_algparams(rem)
    check_invalid(rem)

    a.method = gt.GCP_OPT
    return driver(X, u, a)

def gcp_sgd(X, **kwargs):
    rem = kwargs
    u,rem = make_guess(rem)
    a,rem = make_algparams(rem)
    check_invalid(rem)

    a.method = gt.GCP_SGD
    return driver(X, u, a)
