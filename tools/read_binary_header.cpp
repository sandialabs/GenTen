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

#include "Genten_TensorIO.hpp"



void read_tensor_file(const std::string& filename,
                      std::string& format, std::string& type, bool gz,
                      Genten::Sptensor& x_sparse, Genten::Tensor& x_dense)
{
  Genten::TensorReader<Genten::DefaultHostExecutionSpace> reader(filename,0,gz);
  reader.read();

  if (reader.isSparse()) {
    format = "sparse";
    x_sparse = reader.getSparseTensor();
  }
  else if (reader.isDense()) {
    format = "dense";
    x_dense = reader.getDenseTensor();
  }

  if (reader.isBinary())
    type = "binary";
  else if (reader.isText())
    type = "text";
}

int main(int argc, char* argv[])
{
  int ret = 0;

  auto args = Genten::build_arg_list(argc,argv);
  const bool help =
    Genten::parse_ttb_bool(args, "--help", "--no-help", false);
  if (argc != 3 || help) {
    std::cout << "\ntensor-header: a helper utility for reading headers of binary tensor files.\n\n"
              << "Usage: " << argv[0] << " --input-file <string>\n";
    return 0;
  }

  try {
    Kokkos::initialize(argc, argv);

    const std::string filename =
      Genten::parse_string(args, "--input-file", "");

    Genten::TensorReader<Genten::DefaultHostExecutionSpace> reader(filename);
    if (!reader.isBinary())
      Genten::error("Can only read headers of binary files!");

    std::cout << "\nTensor file:  " << filename << std::endl;
    if (reader.isSparse())
      std::cout << reader.readBinarySparseHeader() << std::endl;
    if (reader.isDense())
      std::cout << reader.readBinaryDenseHeader() << std::endl;

  }
  catch(const std::exception& e)
  {
    std::cout << "*** Call to genten threw an exception:" << std::endl
              << "  " << e.what() << std::endl;
    ret = -1;
  }
  catch(const std::string& s)
  {
    std::cout << "*** Call to genten threw an exception:" << std::endl
              << "  " << s << std::endl;
    ret = -1;
  }
  catch(...)
  {
    std::cout << "*** Call to genten threw an unknown exception"
              << std::endl;
    ret = -1;
  }

  Kokkos::finalize();
  return ret;
}
