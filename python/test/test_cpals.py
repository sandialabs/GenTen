import pygenten as gt
import pytest
import json
from inspect import signature

try:
    import pyttb as ttb
    have_ttb = True
except:
    have_ttb = False

def test_cpals_dense():
    x = gt.import_tensor("data/aminoacid_data_dense.txt")

    u,perf = gt.cp_als(x, rank=16, maxiters=20, tol=1e-4, seed=12345)
    e = perf.lastEntry()

    assert pytest.approx(0.98, abs=0.1) == e.fit
    assert pytest.approx(8, abs=1) == e.iteration

    del u, x

def test_cpals_sparse():
    x = gt.import_sptensor("data/aminoacid_data.txt")

    u,perf = gt.cp_als(x, rank=16, maxiters=20, tol=1e-4, seed=12345)
    e = perf.lastEntry()

    assert pytest.approx(0.98, abs=0.1) == e.fit
    assert pytest.approx(8, abs=1) == e.iteration

    del u, x

def test_cpals_dict():
    x = gt.import_tensor("data/aminoacid_data_dense.txt")
    with open("data/aminoacid-cpals-dense.json",'r') as f:
        params = json.load(f);

    u,perf = gt.cp_als(x, dict=params)
    e = perf.lastEntry()

    assert pytest.approx(params["testing"]["final-fit"]["value"], abs=params["testing"]["final-fit"]["absolute-tolerance"]) == e.fit
    assert pytest.approx(params["testing"]["iterations"]["value"], abs=params["testing"]["iterations"]["absolute-tolerance"]) == e.iteration

    del u, x

def test_cpals_json():
    x = gt.import_tensor("data/aminoacid_data_dense.txt")

    u,perf = gt.cp_als(x, json="data/aminoacid-cpals-dense.json")
    e = perf.lastEntry()

    with open("data/aminoacid-cpals-dense.json",'r') as f:
        params = json.load(f);
    assert pytest.approx(params["testing"]["final-fit"]["value"], abs=params["testing"]["final-fit"]["absolute-tolerance"]) == e.fit
    assert pytest.approx(params["testing"]["iterations"]["value"], abs=params["testing"]["iterations"]["absolute-tolerance"]) == e.iteration

    del u, x

if have_ttb:
    def test_cpals_dense_ttb():
        x = ttb.import_data("data/aminoacid_data_dense.txt")

        u,perf = gt.cp_als(x, rank=16, maxiters=20, tol=1e-4, seed=12345)
        e = perf.lastEntry()

        assert pytest.approx(0.98, abs=0.1) == e.fit
        assert pytest.approx(8, abs=1) == e.iteration

        del u, x

    def test_cpals_sparse_ttb():
        # AminoAcid sparse tensor uses 0-based indexing, which pyttb may not
        # (yet) support
        sig = signature(ttb.import_data)
        if 'index_base' in sig.parameters.keys():
            x = ttb.import_data("data/aminoacid_data.txt", index_base=0)
        else:
            x = gt.make_ttb_sptensor(gt.import_sptensor("data/aminoacid_data.txt"))

        u,perf = gt.cp_als(x, rank=16, maxiters=20, tol=1e-4, seed=12345)
        e = perf.lastEntry()

        assert pytest.approx(0.98, abs=0.1) == e.fit
        assert pytest.approx(8, abs=1) == e.iteration

        del u, x
