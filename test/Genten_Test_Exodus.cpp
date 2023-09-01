//@HEADER
// ************************************************************************
//     Genten: Software for Generalized Tensor Decompositions
//     by Sandia National Laboratories
//
// Sandia National Laboratories is a multimission laboratory managed
// and operated by National Technology and Engineering Solutions of Sandia,
// LLC, a wholly owned subsidiary of Honeywell International, Inc., for the
// U.S. Department of Energy's National Nuclear Security Administration under
// contract DE-NA0003525.
//
// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
// Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// ************************************************************************
//@HEADER

#include <Genten_Tensor.hpp>
#include <Genten_TensorIO.hpp>
#include <Genten_Sptensor.hpp>
#include <Genten_Util.hpp>

#include "Genten_Test_Utils.hpp"

#include <exodusII.h>
#include <json.hpp>
#include <gtest/gtest.h>

namespace Genten {
namespace UnitTests {

template <typename ExecSpace> struct TestExodusT : public ::testing::Test {
  using exec_space = ExecSpace;
};

TYPED_TEST_SUITE(TestExodusT, genten_test_types);

// create a small exodus file with the minimum required data and build a tensor out of it
TYPED_TEST(TestExodusT, TensorFromExodus) {
  using exec_space = typename TestFixture::exec_space;
  using host_exec_space = DefaultHostExecutionSpace;
  int CPU_word_size = sizeof(double);
  int IO_word_size  = sizeof(double);

  // create an exodus file
  std::string base_filename = "test-" + (std::string) exec_space::name();
  std::string exo_filename  = base_filename + ".exo";;
  int exoid = ex_create(exo_filename.c_str(), EX_CLOBBER, &CPU_word_size, &IO_word_size);
  ASSERT_TRUE(exoid > -1);

  // initialize a 2D triangle mesh
  int error = ex_put_init(exoid,
       		          "test" /*title*/,
		          2 /*num_dim*/,
		          4 /*num_nodes*/, 
		          1 /*num_elem*/,
		          1 /*num_elem_blk*/,
		          0 /*num_node_sets*/,
                          0 /*num_side_sets*/);
  ASSERT_EQ(error,0);

  // add coordinates to the mesh
  double x[4] = {0.0, 1.0, 1.0, 0.0};
  double y[4] = {0.0, 0.0, 1.0, 1.0};
  error = ex_put_coord(exoid, x, y, 0);
  ASSERT_EQ(error,0);

  // add coordinate names
  const char *coord_names[] = {"xcoor", "ycoor"};
  error = ex_put_coord_names(exoid, (char **)coord_names);
  ASSERT_EQ(error,0);

  // add element block
  error = ex_put_block(exoid,
		       EX_ELEM_BLOCK,
		       0 /*eblock_id*/,
		       "quad",
		       1 /*num_elem_in_block*/,
                       4 /*num_nodes_per_elem*/,
		       0, 0, 0);
  ASSERT_EQ(error,0);

  // add timesteps
  double times[2] = {0.0, 1.0};

  // add nodal variables
  error = ex_put_variable_param(exoid, EX_NODAL, 2 /*num_nodal_vars*/); 
  ASSERT_EQ(error,0);
  char* var_names[2] = {(char*) "variable1", (char*) "variable2"};
  error = ex_put_variable_names(exoid, EX_NODAL, 2 /*num_nodal_vars*/, var_names); 
  ASSERT_EQ(error,0);

  // add values at each node, for each variable, at each time step
  double vals[2][2][4] = {{{0.0, 1.0, 2.0, 3.0},   {4.0, 5.0, 6.0, 7.0}},
                          {{8.0, 9.0, 10.0, 11.0}, {12.0, 13.0, 14.0, 15.0}}}; 
  double time = 0.0;
  for(int itime = 0; itime < 2; itime++)
  {
    error = ex_put_time(exoid, itime+1, &time);
    ASSERT_EQ(error,0);
    for(int ivar = 0; ivar < 2; ivar++)
    {
      error = ex_put_var(exoid, itime+1, EX_NODAL, ivar+1, 0 /*obj_id*/, 4, vals[itime][ivar]);
      ASSERT_EQ(error,0);
    }
    time = time + 1.0;
  }
  ex_close(exoid);

  // read the exodus file to tensor
  ptree exo_tree;
  exo_tree.add<std::string>("file-type","exodus");
  exo_tree.add<std::string>("format","dense");
  TensorReader<host_exec_space> exo_reader(exo_filename, 0, false, exo_tree);
  exo_reader.read();
  auto exo_tensor = exo_reader.getDenseTensor();

  // write a binary file equivalent to the exodus file
  std::string bin_filename = base_filename + ".bin";
  std::ofstream stream(bin_filename, std::ios::out | std::ios::binary);
  for(int itime = 0; itime < 2; itime++)
    for(int ivar = 0; ivar < 2; ivar++)
      for(int inode = 0; inode < 4; inode++)
      {
        double val = vals[itime][ivar][inode];
        stream.write(reinterpret_cast<const char*>( &val ), sizeof( double ));
      }
  stream.close();

  // read the binary file to tensor
  ptree bin_tree;
  bin_tree.add<std::string>("file-type","binary");
  bin_tree.add<std::string>("format","dense");
  std::vector<int> dims = {4, 2, 2};
  bin_tree.add<std::vector<int>>("dims", dims);
  TensorReader<host_exec_space> bin_reader(bin_filename, 0, false, bin_tree);
  bin_reader.read();
  auto bin_tensor = bin_reader.getDenseTensor();

  // check that the tensors have the correct values in them
  IndxArray index(3);
  for(int itime = 0; itime < 2; itime++)
  {
    index[2] = itime;
    for(int ivar = 0; ivar < 2; ivar++)
    {
      index[1] = ivar; 
      for(int inode = 0; inode < 4; inode++)
      {
	index[0] = inode;
	ASSERT_FLOAT_EQ(exo_tensor[index], vals[itime][ivar][inode]);
	ASSERT_FLOAT_EQ(bin_tensor[index], vals[itime][ivar][inode]);
      }
    }
  }
}

} // namespace UnitTests
} // namespace Genten
