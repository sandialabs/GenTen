from PyGenten import pyGenten

try:
    from matplotlib import pyplot
    have_matplotlib = True
except:
    have_matplotlib = False

pyGenten.initializeKokkos()

x = pyGenten.import_tensor("aminoacid_data_dense.txt")

u,perf = pyGenten.cp_als(x, rank=8, maxiters=20, tol=1e-4)

pyGenten.export_ktensor("aminoacid.ktn", u)

if have_matplotlib:
    it = []
    fit = []
    s = perf.size()
    for i in range(s):
        it.append(perf[i].iteration)
        fit.append(perf[i].fit)

    pyplot.plot(it, fit)
    pyplot.xlabel("iteration")
    pyplot.ylabel("fit")
    pyplot.draw()
    pyplot.savefig("conv.pdf")

del u, x
pyGenten.finalizeKokkos()
