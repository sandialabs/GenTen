import pyttb as ttb
import numpy as np

x = ttb.import_data("data/aminoacid_data.txt", index_base=0)

u,u0,out = ttb.cp_als(x, 16, maxiters=20, stoptol=1e-4)

ttb.export_data(u, "aminoacid.ktn")
