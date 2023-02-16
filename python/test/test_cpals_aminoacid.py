from PyGenten import pyGenten

pyGenten.initializeKokkos()

x = pyGenten.import_tensor("aminoacid_data_dense.txt")

u0 = pyGenten.Ktensor(16, x.ndims(), x.size())
u0.setWeights(1.)
u0.setMatricesRand()

algParams = pyGenten.AlgParams()
algParams.maxiters = 20
algParams.tol = 1.0e-4

u = pyGenten.cpals(x, u0, algParams)

pyGenten.export_ktensor("aminoacid.ktn", u)

del u, u0, x
pyGenten.finalizeKokkos()
