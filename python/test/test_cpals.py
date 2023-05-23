import pygenten as gt
import pytest

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

if have_ttb:
    def test_cpals_dense_ttb():
        x = ttb.import_data("data/aminoacid_data_dense.txt")

        u,perf = gt.cp_als(x, rank=16, maxiters=20, tol=1e-4, seed=12345)
        e = perf.lastEntry()

        assert pytest.approx(0.98, abs=0.1) == e.fit
        assert pytest.approx(8, abs=1) == e.iteration

        del u, x

    def test_cpals_sparse_ttb():
        x = ttb.import_data("data/aminoacid_data.txt", index_base=0)

        u,perf = gt.cp_als(x, rank=16, maxiters=20, tol=1e-4, seed=12345)
        e = perf.lastEntry()

        assert pytest.approx(0.98, abs=0.1) == e.fit
        assert pytest.approx(8, abs=1) == e.iteration

        del u, x
