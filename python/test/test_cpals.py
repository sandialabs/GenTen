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
numIters = 10
resNorm = 1.
perfIter = 10
perfInfo = pyGenten.PerfHistory()

pyGenten.cpals(x, u, algParams, numIters, resNorm, perfIter, perfInfo)

del sizes_np, sizes, u, x
pyGenten.finalizeKokkos()
