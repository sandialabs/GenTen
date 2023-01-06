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

uint64_t smallestBuiltinThatHolds(uint64_t val) {
  if (val <= uint64_t(std::numeric_limits<uint16_t>::max())) {
    return 16;
  }
  if (val <= uint64_t(std::numeric_limits<uint32_t>::max())) {
    return 32;
  }
  return 64; // We didn't have a better option
}

void writeSubValue(std::ostream& outFile, const uint64_t value,
                   const uint64_t size) {
  uint16_t i16;
  uint32_t i32;
  uint64_t i64;
  switch (size) {
  case 16:
    i16 = value;
    outFile.write(reinterpret_cast<char *>(&i16), sizeof(uint16_t));
    break;
  case 32:
    i32 = value;
    outFile.write(reinterpret_cast<char *>(&i32), sizeof(uint32_t));
    break;
  default:
    i64 = value;
    outFile.write(reinterpret_cast<char *>(&i64), sizeof(uint64_t));
  }
}

uint64_t readSubValue(std::istream& inFile, const uint64_t size) {
  uint16_t i16;
  uint32_t i32;
  uint64_t i64;
  switch (size) {
  case 16:
    inFile.read(reinterpret_cast<char *>(&i16), sizeof(uint16_t));
    i64 = i16;
    break;
  case 32:
    inFile.read(reinterpret_cast<char *>(&i32), sizeof(uint32_t));
    i64 = i32;
    break;
  default:
    inFile.read(reinterpret_cast<char *>(&i64), sizeof(uint64_t));
  }
  return i64;
}

void writeDataValue(std::ostream& outFile, const double value,
                    const uint64_t size) {
  float fp32;
  double fp64;
  switch (size) {
  case 16:
    Genten::error("fp16 support not yet implemented");
    // fp16 = value;
    // outFile.write(reinterpret_cast<char*>(&fp16), sizeof(_Float16));
  case 32:
    fp32 = value;
    outFile.write(reinterpret_cast<char*>(&fp32), sizeof(float));
    break;
  default:
    fp64 = value;
    outFile.write(reinterpret_cast<char*>(&fp64), sizeof(double));
  }
}

double readDataValue(std::istream& inFile, const uint64_t size) {
  float fp32;
  double fp64;
  switch (size) {
  case 16:
    Genten::error("fp16 support not yet implemented");
    // inFile.read(reinterpret_cast<char *>(&fp16), sizeof(_Float16));
    // fp64 = fp16;
    break;
  case 32:
    inFile.read(reinterpret_cast<char*>(&fp32), sizeof(float));
    fp64 = fp32;
    break;
  default:
    inFile.read(reinterpret_cast<char*>(&fp64), sizeof(double));
  }
  return fp64;
}

using nd_type = uint32_t;
using nnz_type = uint64_t;
using dim_type = uint64_t;
using sub_size_type = uint64_t;
using float_size_type = uint32_t;

void write_binary_sparse_tensor(const std::string filename,
                                const Genten::Sptensor& x,
                                float_size_type float_data_size = 64)
{
  /*
   * The output file will have the following form:
   * 73 70 74 6e                   -> 4 char 'sptn'
   * ndims                         -> uint32_t
   * bits_for_float_type           -> uint32_t
   * size0 size1 size2 size3 size4 -> ndims uint64_t
   * bits0 bits1 bits2 bits3 bits4 -> number of bits used for each index
   * number_non_zero               -> uint64_t
   * the elements depend on the size of each mode to make the file size smaller
   * we will use the smallest of 8-64 bit unsigned integer that holds all
   * the elements from the size field above, for now all floats are stored as
   * described above.  unlike the textual format we will always use zero based
   * indexing
   * 1 1 1 1049 156 1.000000 -> uint16_t uint16_t uint16_t uint16_t uint32_t
   * float_type
   */
  nd_type nd = x.ndims();
  nnz_type nnz = x.nnz();
  std::vector<sub_size_type> sub_sizes(nd);
  for (auto i=0; i<nd; ++i)
    sub_sizes[i] = smallestBuiltinThatHolds(x.size(i));

  std::ofstream outfile(filename, std::ios::binary);
  if (!outfile)
    Genten::error("Could not open output file " + filename);
  outfile.write("sptn", 4);
  outfile.write(reinterpret_cast<char*>(&nd), sizeof(nd_type));
  outfile.write(reinterpret_cast<char*>(&float_data_size),
                sizeof(float_size_type));
  for (auto n=0; n<nd; ++n) {
    dim_type value = x.size(n);
    outfile.write(reinterpret_cast<char*>(&value), sizeof(dim_type));
  }
  for (auto n=0; n<nd; ++n) {
    auto value = sub_sizes[n];
    outfile.write(reinterpret_cast<char*>(&value), sizeof(sub_size_type));
  }
  outfile.write(reinterpret_cast<char*>(&(nnz)), sizeof(nnz_type));
  for (auto i=0; i<nnz; ++i) {
    for (auto n=0; n<nd; ++n)
      writeSubValue(outfile, x.subscript(i,n), sub_sizes[n]);
    writeDataValue(outfile, x.value(i), float_data_size);
  }
}

Genten::Sptensor read_binary_sparse_tensor(const std::string filename)
{
  std::ifstream infile(filename, std::ios::binary);
  if (!infile)
    Genten::error("Could not open input file " + filename);

  std::string hi = "xxxx";
  infile.read(&hi[0], 4 * sizeof(char));
  if (hi != "sptn")
    Genten::error("First 4 bytes are not sptn");

  // Number of dimensions
  nd_type nd;
  infile.read(reinterpret_cast<char*>(&nd), sizeof(nd_type));

  // Floating point size
  float_size_type float_data_size;
  infile.read(reinterpret_cast<char*>(&float_data_size),
              sizeof(float_size_type));

  // Size of each dimension
  Genten::IndxArray sz(nd);
  for (auto n=0; n<nd; ++n) {
    dim_type value;
    infile.read(reinterpret_cast<char*>(&value), sizeof(dim_type));
    sz[n] = value;
  }

  // Subscript size
  std::vector<sub_size_type> sub_sizes(nd);
  for (auto n=0; n<nd; ++n) {
    sub_size_type value;
    infile.read(reinterpret_cast<char*>(&value), sizeof(sub_size_type));
    sub_sizes[n] = value;
  }

  // Number of nonzeros
  nnz_type nnz;
  infile.read(reinterpret_cast<char*>(&nnz), sizeof(nnz_type));

  // Allocate tensor
  Genten::Sptensor x(sz, nnz);

  // Nonzeros
  for (auto i=0; i<nnz; ++i) {
    for (auto n=0; n<nd; ++n)
      x.subscript(i,n) = readSubValue(infile, sub_sizes[n]);
    x.value(i) = readDataValue(infile, float_data_size);
  }

  return x;
}

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
    print_tensor_stats(x_out);
    if (type == "text")
      Genten::export_sptensor(filename, x_out);
    else if (type == "binary")
      write_binary_sparse_tensor(filename, x_out);
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
    if (input_type != "text" && input_type != "binary")
      Genten::error("input type must be one of \"text\" or \"binary\"");
    if (output_type != "text" && output_type != "binary")
      Genten::error("output type must be one of \"text\" or \"binary\"");

    std::cout << "\nInput:\n"
              << "  File:   " << input_filename << std::endl
              << "  Format: " << input_format << std::endl
              << "  Type:   " << input_type << std::endl;
    if (input_format == "sparse") {
      Genten::Sptensor x_in;
      if (input_type == "text")
        Genten::import_sptensor(input_filename, x_in);
      else if (input_type == "binary")
        x_in = read_binary_sparse_tensor(input_filename);
      print_tensor_stats(x_in);
      save_tensor(x_in, output_filename, output_format, output_type);
    }
    else if (input_format == "dense") {
      Genten::Tensor x_in;
      if (input_type == "text")
        Genten::import_tensor(input_filename, x_in);
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
