import pyttb as ttb
import numpy as np

x = ttb.import_data("data/aminoacid_data_dense.txt")

u,u0,out = ttb.cp_als(x, 16, maxiters=20, stoptol=1e-4)

ttb.export_data(u, "aminoacid.ktn")
ttb.export_data(u.full(), "aminoacid_reconstruction.tns")
