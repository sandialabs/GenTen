"""
Various utility functions for using GenTen.
"""

import pygenten._pygenten as gt
import json

def make_algparams(args):
    """
    Utility function for converting a dict to a pygenten.AlgParms.
    """
    a = gt.AlgParams()
    rem = {}
    for key,value in args.items():
        if key == "goal":
            a.set_py_goal(value)
        elif key == "proc_grid":
            a.set_proc_grid(value)
        elif key == "dict":
            json_params = json.dumps(value)
            a.parse_json(json_params)
        elif key == "json":
             with open(value,'r') as f:
                 json_params = json.load(f)
                 json_params = json.dumps(json_params)
                 a.parse_json(json_params)
        elif hasattr(a, key):
            setattr(a, key, value)
        else:
            rem[key] = value
    return a, rem

def read_and_distribute_tensor(filename, file_type=None, format=None, shape=None, nnz=None, value_bits=None, sub_bits=None, index_base=0, compressed=False, **kwargs):
    """
    Read a tensor from a file and distribute it in parallel.

    Parameters:
      * filename (str):  name of file to read
      * file_type (str): type of file (test or binary)
      * format (str): tensor format (sparse or dense)
      * shape (tuple): shape of tensor for reading binary files without headers
      * nnz (int): number of tensor nonzeros for reading binary files without
        headers
      * value_bits (int): number of bits used to store tensor values for reading
        binary files without headers
      * sub_bits (int): number of bits used to store tensor nonzero coordinates
        for reading sparse, binary files without headers
      * index_base (int): starting index for nonzero coordinates in the file
      * comopressed (bool): whether the file is compressed or not
      * kwards: keyword arguments for additional parameters needed when reading
        the file.

    Returns the dense/sparse tensor and the distributed tensor context.
    """
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
    if Xs.nnz > 0:
        return Xs,dtc
    return Xd,dtc

def distribute_tensor(X, **kwargs):
    """
    Distribute the given tensor in parallel and return the result.

    Also returns the distributed tensor context.
    """
    a,rem = make_algparams(kwargs)
    dtc = gt.DistTensorContext()
    XX = dtc.distributeTensor(X, a)
    return XX,dtc
