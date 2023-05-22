import pyttb as ttb
import numpy as np

x = ttb.import_data("data/aminoacid_data.txt")

# import_data subtracts 1 from the indices which it shouldn't
for i in range(x.nnz):
    x.subs[i,:] = x.subs[i,:]+1

u,u0,out = ttb.cp_als(x, 16, maxiters=20, stoptol=1e-4)

ttb.export_data(u, "aminoacid.ktn")
