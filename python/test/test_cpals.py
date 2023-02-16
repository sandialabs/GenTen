from PyGenten import pyGenten
import numpy as np

pyGenten.initializeKokkos()
sizes = pyGenten.IndxArray(3)
sizes_np = np.array(sizes, copy=False)
sizes_np[0:3] = [3,4,5]

x = pyGenten.Tensor(sizes, 2.)
u = pyGenten.Ktensor(1, 3, sizes)
u.setWeights(1.)
u.setMatricesRand()

algParams = pyGenten.AlgParams()

u = pyGenten.cpals(x, u, algParams)

del sizes_np, sizes, u, x
pyGenten.finalizeKokkos()
