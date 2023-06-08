import pygenten

try:
    from matplotlib import pyplot
    have_matplotlib = True
except:
    have_matplotlib = False

x = pygenten.import_sptensor("data/aminoacid_data.txt")

u,perf = pygenten.cp_als(x, rank=8, maxiters=20, tol=1e-4, timings=True)

if pygenten.proc_rank() == 0:
    pygenten.export_ktensor("aminoacid.ktn", u)

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
