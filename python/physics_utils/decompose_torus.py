# example usage:
#   decomp --processors [num_procs] --subdir exo_files exo_files/test_larger_kappa_perp.exo
#   mpiexec -n [num_procs] python3 decompose_torus.py exo_files/test_larger_kappa_perp.exo z

import os
import sys
import exodus3 as ex
import numpy as np
import pygenten as gt
import pyttb
import torus_to_tensor

if __name__ == "__main__":
   assert len(sys.argv) >= 3, "usage: decompose_torus.py <exodus_base_name> <axis_of_rotation>"
   num_procs = gt.num_procs()
   base_filename = sys.argv[1]
   axis = sys.argv[2]
   tol = 1.0e-10
   rescale_variables = True
   if len(sys.argv) >= 4:
     tol = np.double(sys.argv[3])

   num_procs_per_poloidal_plane = num_procs
   split_filename = base_filename.split("/")
   new_filename = ""
   new_filenamex = ""
   for i in range(len(split_filename)-1):
     new_filename = new_filename + split_filename[i] + "/"
     new_filenamex = new_filenamex + split_filename[i] + "/"
   new_filename = new_filename + "reconstructed_" + split_filename[-1]
   new_filenamex = new_filenamex + "test_" + split_filename[-1]

   gt_dist_tensor, gt_dtc, tensor_node_gids, mt, st = torus_to_tensor.torus_to_tensor(base_filename, axis, num_procs_per_poloidal_plane, tol, rescale_variables)
   u,perf = gt.cp_als(gt_dist_tensor, dtc=gt_dtc, rank=16, maxiters=200, tol=1e-4, seed=12345, dist_guess_method="parallel")

   unscaled_data = u.full().data * st + mt
   ttb_tensor = pyttb.tensor.from_data(unscaled_data)
   gt_tensor = gt.make_gt_tensor(ttb_tensor)
   torus_to_tensor.tensor_to_exodus(gt_tensor, tensor_node_gids, base_filename, new_filename)

   del(gt_tensor)
   del(u)
   del(perf)
   del(gt_dist_tensor)




