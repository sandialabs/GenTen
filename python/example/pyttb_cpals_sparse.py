import pyttb as ttb
import numpy as np
import pygenten

if pygenten.proc_rank() == 0:
    x = ttb.import_data("data/aminoacid_data.txt", index_base=0)
else:
    x = ttb.sptensor()

#u,u0,out = ttb.cp_als(x, 8, maxiters=20, stoptol=1e-4)
u,_ = pygenten.cp_als(x, rank=8, maxiters=20, tol=1e-4)

if pygenten.proc_rank() == 0:
    ttb.export_data(u, "aminoacid.ktn")

del x,u
