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

def cp_als(X, **kwargs):
    rem = kwargs
    u,rem = make_guess(rem)
    a,rem = make_algparams(rem)
    check_invalid(rem)

    a.method = gt.CP_ALS
    return gt.driver(X, u, a)

def cp_opt(X, **kwargs):
    rem = kwargs
    u,rem = make_guess(rem)
    a,rem = make_algparams(rem)
    check_invalid(rem)

    a.method = gt.CP_OPT
    return gt.driver(X, u, a)

def gcp_opt(X, **kwargs):
    rem = kwargs
    u,rem = make_guess(rem)
    a,rem = make_algparams(rem)
    check_invalid(rem)

    a.method = gt.GCP_OPT
    return gt.driver(X, u, a)

def gcp_sgd(X, **kwargs):
    rem = kwargs
    u,rem = make_guess(rem)
    a,rem = make_algparams(rem)
    check_invalid(rem)

    a.method = gt.GCP_SGD
    return gt.driver(X, u, a)
