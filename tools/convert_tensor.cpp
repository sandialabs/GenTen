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
                 const std::string format, const std::string type, bool gz,
                 bool header)
{
  std::cout << "\nOutput:\n"
            << "  File:   " << filename << std::endl
            << "  Format: " << format << std::endl
            << "  Type:   " << type;
  if (type == "text" && gz)
    std::cout << " (compressed)";
  if (type == "binary" && !header)
    std::cout << " (no header)";
  std::cout << std::endl;
  Genten::TensorWriter<Genten::DefaultHostExecutionSpace> writer(filename,gz);
  if (format == "sparse") {
    Genten::Sptensor x_out(x_in);
    print_tensor_stats(x_out);
    if (type == "text")
      writer.writeText(x_out);
    else if (type == "binary")
      writer.writeBinary(x_out, header);
  }
  else if (format == "dense") {
    Genten::Tensor x_out(x_in);
    print_tensor_stats(x_out);
    if (type == "text")
      writer.writeText(x_out);
    else if (type == "binary")
      writer.writeBinary(x_out, header);
  }
}

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
  if (argc < 9 || argc > 11 || help) {
    std::cout << "\nconvert-tensor: a helper utility for converting tensor data between\n"
              << "tensor formats (sparse or dense), and file types (text or binary).\n\n"
              << "Usage: " << argv[0] << " --input-file <string> --output-file <string> --output-format <sparse|dense> --output-type <text|binary> [options] \n"
              << "Options:\n"
              << "  --input-gz      Input tensor is Gzip compressed (text-only, default: off)\n"
              << "  --output-gz     Output tensor is Gzip compressed (text-only, default: off)\n"
              << "  --output-header Write header to output file (binary-only, default: on)\n";
    return 0;
  }

  try {
    Kokkos::initialize(argc, argv);

    const std::string input_filename =
      Genten::parse_string(args, "--input-file", "");
    const std::string output_filename =
      Genten::parse_string(args, "--output-file", "");
    const std::string output_format =
      Genten::parse_string(args, "--output-format", "");
    const std::string output_type =
      Genten::parse_string(args, "--output-type", "");
    const bool input_gz =
      Genten::parse_ttb_bool(args, "--input-gz", "--no-input-gz", false);
    const bool output_gz =
      Genten::parse_ttb_bool(args, "--output-gz", "--no-output-gz", false);
    const bool output_header =
      Genten::parse_ttb_bool(args, "--output-header", "--no-output-header", true);

    if (input_filename == "")
      Genten::error("input filename must be specified");
    if (output_filename == "")
      Genten::error("output filename must be specified");
    if (output_format != "sparse" && output_format != "dense")
      Genten::error("output format must be one of \"sparse\" or \"dense\"");
    if (output_type != "text" && output_type != "binary")
      Genten::error("output type must be one of \"text\" or \"binary\"");
    if (output_gz && output_type != "text")
      Genten::error("gzip only supported for text-based output files");
    if (!output_header && output_type != "binary")
      Genten::error("No header option only supported for binary output files");

    std::cout << "\nInput:\n"
              << "  File:   " << input_filename << std::endl;

    std::string input_format = "unknown";
    std::string input_type = "unknown";
    Genten::Sptensor x_sparse;
    Genten::Tensor x_dense;
    read_tensor_file(input_filename, input_format, input_type, input_gz,
                     x_sparse, x_dense);

    std::cout << "  Format: " << input_format << std::endl
              << "  Type:   " << input_type;
    if (input_type == "text" && input_gz)
      std::cout << " (compressed)";
    std::cout << std::endl;
    if (input_format == "sparse") {
      print_tensor_stats(x_sparse);
      save_tensor(x_sparse, output_filename, output_format, output_type,
                  output_gz, output_header);
    }
    else if (input_format == "dense") {
      print_tensor_stats(x_dense);
      save_tensor(x_dense, output_filename, output_format, output_type,
                  output_gz, output_header);
    }
    else
      Genten::error("Invalid input tensor format!");

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
