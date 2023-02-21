import pygenten
import numpy as np

pygenten.initializeKokkos()
sizes = pygenten.IndxArray(3)
sizes_np = np.array(sizes, copy=False)
sizes_np[0:3] = [3,4,5]

x = pygenten.Tensor(sizes, 2.)
u = pygenten.Ktensor(1, 3, sizes)
u.setWeights(1.)
u.setMatricesRand()

algParams = pygenten.AlgParams()

u = pygenten.cpals(x, u, algParams)

del sizes_np, sizes, u, x
pygenten.finalizeKokkos()
