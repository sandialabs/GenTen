# example usage:
#   decomp --processors [num_procs] --subdir exo_files exo_files/test_larger_kappa_perp.exo
#   mpiexec -n [num_procs] python3 decompose_torus.py exo_files/test_larger_kappa_perp.exo z

import os
import sys
import exodus3 as ex
import numpy as np
import pygenten as gt
import pyttb
sys.path.append('..')
import torus_to_tensor

if __name__ == "__main__":
   assert len(sys.argv) >= 3, "usage: decompose_torus.py <exodus_base_name> <axis_of_rotation>"
   num_procs = gt.num_procs()
   base_filename = sys.argv[1]
   axis = sys.argv[2]
   tol = 1.0e-10
   if len(sys.argv) >= 4:
     tol = np.double(sys.argv[3])

   num_procs_per_poloidal_plane = num_procs
   np_tensor, global_blocking, parallel_map = torus_to_tensor.torus_to_tensor(base_filename, axis, num_procs_per_poloidal_plane, tol)
   shape = np_tensor.shape
   print("Dimensions on proc " + str(gt.proc_rank()) + ":")
   print("  Num toroidal:  "+str(shape[0]))
   print("  Num poloidal:  "+str(shape[1]))
   print("  Num variables: "+str(shape[2]))
   print("  Num times:     "+str(shape[3]))

   ttb_tensor = pyttb.tensor.from_data(np_tensor)
   gt_tensor = gt.make_gt_tensor(ttb_tensor)

   gt_dist_tensor, gt_dtc = gt.distribute_tensor(gt_tensor, global_blocking, parallel_map)
   del(ttb_tensor)
   del(gt_tensor)
   u,perf = gt.cp_als(gt_dist_tensor, dtc=gt_dtc, rank=16, maxiters=200, tol=1e-4, seed=12345, dist_guess_method="parallel")
   del(u)
   del(perf)
   del(gt_dist_tensor)




