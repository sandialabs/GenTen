import pygenten._pygenten as gt
import json

def make_algparams(args):
    a = gt.AlgParams()
    rem = {}
    for key,value in args.items():
        if key == "goal":
            a.set_py_goal(value)
        elif key == "proc_grid":
            a.set_proc_grid(value)
        elif hasattr(a, key):
            setattr(a, key, value)
        else:
            rem[key] = value
    return a, rem

def read_and_distribute_tensor(filename, file_type=None, format=None, shape=None, nnz=None, value_bits=None, sub_bits=None, index_base=0, compressed=False, **kwargs):
    a,rem = make_algparams(kwargs)
    d = dict()
    if format is not None:
        d["format"] = format
    if file_type is not None:
        d["file-type"] = file_type
    if shape is not None:
        d["dims"] = list(shape)
    if nnz is not None:
        d["nnz"] = nnz
    if value_bits is not None:
        d["value-bits"] = value_bits
    if sub_bits is not None:
        d["sub-bits"] = sub_bits
    j = json.dumps(d)
    dtc = gt.DistTensorContext()
    Xs,Xd = dtc.distributeTensor(filename, index_base, compressed, j, a)
    if Xs.nnz() > 0:
        return Xs,dtc
    return Xd,dtc

def distribute_tensor(X, **kwargs):
    a,rem = make_algparams(kwargs)
    dtc = gt.DistTensorContext()
    XX = dtc.distributeTensor(X, a)
    return XX,dtc
