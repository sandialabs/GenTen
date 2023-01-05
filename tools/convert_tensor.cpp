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

#include "Genten_Sptensor.hpp"
#include "Genten_Tensor.hpp"
#include "Genten_IOtext.hpp"

template <typename TensorType>
void print_tensor_stats(const TensorType& x)
{
  std::cout << "  Stats:  ";
  const ttb_indx nd = x.ndims();
  for (ttb_indx i=0; i<nd; ++i) {
    std::cout << x.size(i);
    if (i != nd-1)
      std::cout << " x ";
  }
  const ttb_indx nnz = x.nnz();
  const ttb_real ne = x.numel_float();
  const ttb_real nz = ne - nnz;
  std::cout << ", " << std::setprecision(0) << std::fixed << ne
            << " entries, " << nnz << " ("
            << std::setprecision(1) << std::fixed
            << 100.0*(nnz/ne)
            << "%) nonzeros and "
            << std::setprecision(0) << std::fixed << nz << " ("
            << std::setprecision(1) << std::fixed << 100.0*(nz/ne)
            << "%) zeros" << std::endl;
}

template <typename TensorType>
void save_tensor(const TensorType& x_in, const std::string& filename,
                 const std::string format, const std::string type)
{
  std::cout << "\nOutput:\n"
              << "  File:   " << filename << std::endl
              << "  Format: " << format << std::endl
              << "  Type:   " << type << std::endl;
  if (format == "sparse") {
    Genten::Sptensor x_out(x_in);
    if (type == "text") {
      Genten::export_sptensor(filename, x_out);
    }
  }
  else if (format == "dense") {
    Genten::Tensor x_out(x_in);
    print_tensor_stats(x_out);
    if (type == "text") {
      Genten::export_tensor(filename, x_out);
    }
  }
}

int main(int argc, char* argv[])
{
  int ret = 0;

  auto args = Genten::build_arg_list(argc,argv);
  const bool help =
    Genten::parse_ttb_bool(args, "--help", "--no-help", false);
  if (argc < 3 || argc > 13 || help) {
    std::cout << "\nconvert-tensor: a helper utility for converting tensor data between\n"
              << "tensor formats (sparse or dense), and file types (text or binary).\n\n"
              << "Usage: " << argv[0] << " --input-file <string> --output-file <string> [options]\n"
              << "\nRequired arguments: \n"
              << "  --input-file <string>          path to input tensor data\n"
              << "  --output-file <string>         path to output tensor data\n"
              << "\nOptions: \n"
              << "  --input-format <sparse|dense>  format of input tensor (default: \"sparse\")\n"
              << "  --input-type <text|binary>     type of input tensor data (default: \"text\")\n"
              << "  --output-format <sparse|dense> format of output tensor (default: \"dense\")\n"
              << "  --output-type <text|binary>    type of output tensor data  (default: \"text\")\n";
    return 0;
  }

  try {
    Kokkos::initialize(argc, argv);

    const std::string input_filename =
      Genten::parse_string(args, "--input-file", "");
    const std::string input_format =
      Genten::parse_string(args, "--input-format", "sparse");
    const std::string input_type =
      Genten::parse_string(args, "--input-type", "text");
    const std::string output_filename =
      Genten::parse_string(args, "--output-file", "");
    const std::string output_format =
      Genten::parse_string(args, "--output-format", "dense");
    const std::string output_type =
      Genten::parse_string(args, "--output-type", "text");

    if (input_filename == "")
      Genten::error("input filename must be specified");
    if (output_filename == "")
      Genten::error("output filename must be specified");
    if (input_format != "sparse" && input_format != "dense")
      Genten::error("input format must be one of \"sparse\" or \"dense\"");
    if (output_format != "sparse" && output_format != "dense")
      Genten::error("output format must be one of \"sparse\" or \"dense\"");
    if (input_type != "text" && input_format != "binary")
      Genten::error("input type must be one of \"text\" or \"binary\"");
    if (output_type != "text" && output_format != "binary")
      Genten::error("output type must be one of \"text\" or \"binary\"");

    std::cout << "\nInput:\n"
              << "  File:   " << input_filename << std::endl
              << "  Format: " << input_format << std::endl
              << "  Type:   " << input_type << std::endl;
    if (input_format == "sparse") {
      Genten::Sptensor x_in;
      if (input_type == "text") {
        Genten::import_sptensor(input_filename, x_in);
      }
      print_tensor_stats(x_in);
      save_tensor(x_in, output_filename, output_format, output_type);
    }
    else if (input_format == "dense") {
      Genten::Tensor x_in;
      if (input_type == "text") {
        Genten::import_tensor(input_filename, x_in);
      }
      print_tensor_stats(x_in);
      save_tensor(x_in, output_filename, output_format, output_type);
    }

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
