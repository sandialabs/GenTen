# documentation

import os
import sys
import exodus3 as ex
import numpy as np
import _pygenten as gt
import _phys_utils as pu
sys.path.append('..')
import torus_to_tensor
import unittest

class TestTorusToTensor(unittest.TestCase):
  def test(self):
    num_times = 1
    setup_exo_files(num_times)

    num_procs_per_poloidal_plane = 2
    tensor = torus_to_tensor.torus_to_tensor("simple_torus.exo",'y',num_procs_per_poloidal_plane)
    test_tensor(self,tensor,num_times,num_procs_per_poloidal_plane)

    num_procs_per_poloidal_plane = 4
    tensor = torus_to_tensor.torus_to_tensor("simple_torus.exo",'y',num_procs_per_poloidal_plane)
    test_tensor(self,tensor,num_times,num_procs_per_poloidal_plane)

    os.remove("simple_torus.exo.4."+str(gt.proc_rank()))

def setup_exo_files(n_times):
    base_filename = "simple_torus.exo.4."
    rank = gt.proc_rank()
   
    # create a new exo file
    exo_filename = base_filename + str(rank)
    if os.path.exists(exo_filename):
      os.remove(exo_filename)
    num_elems = 2 - rank % 2
    ex_pars = ex.ex_init_params(num_dim=3,
                                num_nodes=4*num_elems,
                                num_elem=num_elems,
                                num_elem_blk=1)
    exo_file = ex.exodus(exo_filename, mode="w", array_type='numpy', init_params=ex_pars)

    # add timesteps
    for itime in range(n_times):
      exo_file.put_time(itime+1, 0.1*itime)

    # add some variables
    n_vars = 4
    exo_file.set_node_variable_number(n_vars)
    for ivar in range(n_vars):
      exo_file.put_node_variable_name("nodal_var_"+str(ivar), ivar+1)

    # set up the node map and coordinates
    gids, x, y, z = setup_local_gids_and_coords()

    # add gids, coords, and some values
    # for testing, let the variable values be derived from the gids and x,y,z coordinates scaled by time
    exo_file.put_id_map('EX_NODE_MAP',gids)
    exo_file.put_coords(x,y,z)
    for itime in range(n_times):
      for ivar in range(n_vars):
        data = gids
        if ivar == 1:
          data = x
        elif ivar == 2:
          data = y
        elif ivar == 3:
          data = z
        values = [(i+1)*(itime+1) for i in data]
        exo_file.put_node_variable_values("nodal_var_"+str(ivar),itime+1,values)

    exo_file.close()
   
def setup_local_gids_and_coords():
    # let the mesh be 3 2-element poloidal slices split across four ranks
    # node GIDs are labeled
    #
    # theta = 0
    #   3 --------- 4 --------- 5 y=1
    #   |           |           |
    #   |  rank 0   |  rank 1   |
    #   |           |           |
    #   0 --------- 1 --------- 2 y=0
    #   x=1         x=2         x=3       z=0
    #
    # theta = pi/6
    #   9 --------- 10--------- 11 y=1
    #   |           |           |
    #   |  rank 0   |  rank 2   |
    #   |           |           |
    #   6 --------- 7 --------- 8 y=0
    #   x=sqrt(3)/2 x=sqrt(3)   x=3/2*sqrt(3)
    #   z=1/2       z=1         z=3/2
    #   
    # theta = pi
    #   15--------- 16--------- 17 y=1
    #   |           |           |
    #   |  rank 2   |  rank 3   |
    #   |           |           |
    #   12--------- 13--------- 14 y=0
    #   x=-1       x=-2         x=-3      z=0

    # LIDs per element
    #   3 --- 2
    #   |     |
    #   0 --- 1

    gids, x, y, z = [], [], [], []
    rank = gt.proc_rank()
    if rank == 0:
      gids = [0, 1, 4, 3, 6, 7, 10, 9]
      x = [1.0, 2.0, 2.0, 1.0, np.sqrt(3.0)*0.5, np.sqrt(3.0), np.sqrt(3.0), np.sqrt(3.0)*0.5]
      y = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
      z = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 0.5]
    elif rank == 1:
      gids = [1, 2, 5, 4]
      x = [2.0, 3.0, 3.0, 2.0]
      y = [0.0, 0.0, 1.0, 1.0]
      z = [0.0, 0.0, 0.0, 0.0]
    elif rank == 2:
      gids = [7, 8, 11, 10, 12, 13, 16, 15]
      x = [np.sqrt(3.0), 1.5*np.sqrt(3.0), 1.5*np.sqrt(3.0), np.sqrt(3.0), -1.0, -2.0, -2.0, -1.0]
      y = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
      z = [1.0, 1.5, 1.5, 1.0, 0.0, 0.0, 0.0, 0.0]
    elif rank == 3:
      gids = [13, 14, 17, 16]
      x = [-2.0, -3.0, -3.0, -2.0]
      y = [0.0, 0.0, 1.0, 1.0]
      z = [0.0, 0.0, 0.0, 0.0]
    return gids, x, y, z

def test_tensor(self, tensor, num_times, num_procs_per_poloidal_plane):
    num_procs = gt.num_procs()
    rank = gt.proc_rank()  
    total_poloidal_nodes = 6
    total_thetas = 3
    total_nodes = total_poloidal_nodes*total_thetas
    num_vars = 4
    tol = 1.0e-10

    # check the dimensions of the tensor
    num_procs_per_theta = num_procs//num_procs_per_poloidal_plane
    num_local_thetas = 3 
    if num_procs_per_theta == 2:
      num_local_thetas = 2
      if rank > 1:
        num_local_thetas = 1 

    num_local_poloidal_nodes = 3
    if num_procs_per_poloidal_plane == 4:
      num_local_poloidal_nodes = 2
      if rank > 1:
        num_local_poloidal_nodes = 1 

    self.assertEqual(tensor.shape[0],num_local_thetas) 
    self.assertEqual(tensor.shape[1],num_local_poloidal_nodes) 
    self.assertEqual(tensor.shape[2],num_vars) 
    self.assertEqual(tensor.shape[3],num_times) 

    # entry (i,j,0,0) should be GID of the corresponding node + 1
    # check that we have all GIDs and that they're uniquely assigned to procs
    local_gid_frequency = np.zeros(total_nodes, dtype=int)
    num_local_nodes = num_local_thetas*num_local_poloidal_nodes
    local_gids = -1*np.ones(num_local_nodes, dtype=np.longlong)
    for i in range(num_local_thetas):
      for j in range(num_local_poloidal_nodes):
        gid = np.longlong(tensor[i,j,0,0] - 1)
        local_gid_frequency[gid] = local_gid_frequency[gid] + 1
        local_gids[i*num_local_poloidal_nodes+j] = gid
    global_gid_frequency = pu.global_int_array_sum(local_gid_frequency)
    for i in range(total_nodes):
      self.assertEqual(global_gid_frequency[i],1)

    # make sure that all nodes with same theta index have the same theta value
    # same with poloidal nodes and (r,y) coordinates
    thetas = -1*np.ones(total_thetas, dtype=np.double)
    rs     = -1*np.ones(total_poloidal_nodes, dtype=np.double)
    ys     = -1*np.ones(total_poloidal_nodes, dtype=np.double)
    for i in range(num_local_thetas):
      global_theta_index = i
      if num_procs_per_theta == 2 and rank > 1:
        global_theta_index += 2
      for j in range(num_local_poloidal_nodes):
        global_poloidal_index = j
        if num_procs_per_poloidal_plane == 2:
          if rank%2 > 0:
            global_poloidal_index += 3
        elif num_procs_per_poloidal_plane == 4:
          if rank==1:
            global_poloidal_index += 2
          elif rank==2:
            global_poloidal_index += 4
          elif rank==3:
            global_poloidal_index += 5
        x = tensor[i,j,1,0] - 1.0
        y = tensor[i,j,2,0] - 1.0
        z = tensor[i,j,3,0] - 1.0
        r = np.sqrt(x*x+z*z)
        theta = np.arccos(x/r)
        if thetas[global_theta_index] < -0.5:
          thetas[global_theta_index] = theta
        else:
          self.assertTrue(np.abs(thetas[global_theta_index] - theta) < tol)
        if rs[global_poloidal_index] < -0.5:
          rs[global_poloidal_index] = r
          ys[global_poloidal_index] = y
        else:
          self.assertTrue(np.abs(rs[global_poloidal_index] - r) < tol)
          self.assertTrue(np.abs(ys[global_poloidal_index] - y) < tol)

    # confirm across processors
    for p in range(num_procs):
      # broadcast
      remote_thetas = np.zeros(total_thetas, dtype=np.double)
      remote_rs     = np.zeros(total_poloidal_nodes, dtype=np.double)
      remote_ys     = np.zeros(total_poloidal_nodes, dtype=np.double)
      if rank == p:
        remote_thetas = thetas
        remote_rs = rs
        remote_ys = ys
      pu.broadcast_scalar_data(p, remote_thetas)
      pu.broadcast_scalar_data(p, remote_rs)
      pu.broadcast_scalar_data(p, remote_ys)
      for i in range(total_thetas):
        if remote_thetas[i] < -0.5 or thetas[i] < -0.5:
          continue
        else:
          self.assertTrue(np.abs(remote_thetas[i] - thetas[i]) < tol)
      for i in range(total_poloidal_nodes):
        if remote_rs[i] < -0.5 or rs[i] < -0.5:
          continue
        else:
          self.assertTrue(np.abs(remote_rs[i] - rs[i]) < tol)
          self.assertTrue(np.abs(remote_ys[i] - ys[i]) < tol)

    
if __name__ == "__main__":
    gt.initializeGenten()
    unittest.main()
    gt.finalizeGenten()
