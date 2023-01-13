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
#include "Genten_IOtext.hpp"

namespace {

using nd_type = uint32_t;
using nnz_type = uint64_t;
using dim_type = uint64_t;
using sub_size_type = uint64_t;
using float_size_type = uint32_t;

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

void write_binary_dense_tensor(const std::string filename,
                               const Genten::Tensor& x,
                               float_size_type float_data_size = 64)
{
  /*
   * The output file will have the following form:
   * 73 70 74 6e                   -> 4 char 'dntn'
   * ndims                         -> uint32_t
   * bits_for_float_type           -> uint32_t
   * size0 size1 size2 size3 size4 -> ndims uint64_t
   * number_non_zero               -> uint64_t
   * 1.000000                      -> float_type
   */
  nd_type nd = x.ndims();
  nnz_type ne = x.numel();

  std::ofstream outfile(filename, std::ios::binary);
  if (!outfile)
    Genten::error("Could not open output file " + filename);
  outfile.write("dntn", 4);
  outfile.write(reinterpret_cast<char*>(&nd), sizeof(nd_type));
  outfile.write(reinterpret_cast<char*>(&float_data_size),
                sizeof(float_size_type));
  for (auto n=0; n<nd; ++n) {
    dim_type value = x.size(n);
    outfile.write(reinterpret_cast<char*>(&value), sizeof(dim_type));
  }
  outfile.write(reinterpret_cast<char*>(&(ne)), sizeof(nnz_type));
  for (auto i=0; i<ne; ++i) {
    writeDataValue(outfile, x[i], float_data_size);
  }
}

Genten::Sptensor read_binary_dense_tensor(const std::string filename)
{
  std::ifstream infile(filename, std::ios::binary);
  if (!infile)
    Genten::error("Could not open input file " + filename);

  std::string hi = "xxxx";
  infile.read(&hi[0], 4 * sizeof(char));
  if (hi != "dntn")
    Genten::error("First 4 bytes are not dntn");

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

  // Number of elements
  nnz_type ne;
  infile.read(reinterpret_cast<char*>(&ne), sizeof(nnz_type));

  // Allocate tensor
  Genten::Tensor x(sz);

  // Nonzeros
  for (auto i=0; i<ne; ++i)
    x[i] = readDataValue(infile, float_data_size);

  return x;
}

}

namespace Genten {

template <typename ExecSpace>
TensorReader<ExecSpace>::
TensorReader(const std::string& filename,
             const ttb_indx index_base,
             const bool compressed) :
  is_sparse(false), is_dense(false), is_binary(false), is_text(false)
{
  // First try reading the tensor as a binary file
  {
    std::ifstream file(filename, std::ios::binary);
    if (!file)
      Genten::error("Cannot open input file: " + filename);
    try {
      std::string header = "xxxx";
      file.read(&header[0], 4);
      if (header == "sptn") {
        Sptensor X = read_binary_sparse_tensor(filename);
        X_sparse = create_mirror_view(ExecSpace(), X);
        deep_copy(X_sparse, X);
        is_sparse = true;
        is_binary = true;
        return;
      }
      if (header == "dntn") {
        Tensor X = read_binary_dense_tensor(filename);
        X_dense = create_mirror_view(ExecSpace(), X);
        deep_copy(X_dense, X);
        is_dense = true;
        is_binary = true;
        return;
      }
    } catch (...) {}
  }

  // If that failed, try reading it as text
  {
    std::ifstream file(filename);
    if (!file)
      Genten::error("Cannot open input file: " + filename);
    try {
      std::string line;
      std::getline(file, line);
      if (line == "tensor") {
        Tensor X;
        Genten::import_tensor(filename, X);
        X_dense = create_mirror_view(ExecSpace(), X);
        deep_copy(X_dense, X);
        is_dense = true;
        is_text = true;
        return;
      }
      // We support sparse tensor files without a header, so just try reading
      // it as a sparse tensor if we have gotten this far.  It will throw
      // if that read fails.
      Sptensor X;
      Genten::import_sptensor(filename, X, index_base, compressed);
      X_sparse = create_mirror_view(ExecSpace(), X);
      deep_copy(X_sparse, X);
      is_sparse = true;
      is_text = true;
    } catch (...) {
      Genten::error("File " + filename + " cannot be read as a text or binary, sparse or dense tensor!");
    }
  }
}

template <typename ExecSpace>
TensorWriter<ExecSpace>::
TensorWriter(const std::string& fname) : filename(fname) {}

template <typename ExecSpace>
void
TensorWriter<ExecSpace>::
writeBinary(const SptensorT<ExecSpace>& X) const
{
  Sptensor X_host = create_mirror_view(X);
  deep_copy(X_host, X);
  write_binary_sparse_tensor(filename, X_host);
}

template <typename ExecSpace>
void
TensorWriter<ExecSpace>::
writeBinary(const TensorT<ExecSpace>& X) const
{
  Tensor X_host = create_mirror_view(X);
  deep_copy(X_host, X);
  write_binary_dense_tensor(filename, X_host);
}

template <typename ExecSpace>
void
TensorWriter<ExecSpace>::
writeText(const SptensorT<ExecSpace>& X) const
{
  Sptensor X_host = create_mirror_view(X);
  deep_copy(X_host, X);
  export_sptensor(filename, X_host);
}

template <typename ExecSpace>
void
TensorWriter<ExecSpace>::
writeText(const TensorT<ExecSpace>& X) const
{
  Tensor X_host = create_mirror_view(X);
  deep_copy(X_host, X);
  export_tensor(filename, X_host);
}

}

#define INST_MACRO(SPACE) \
  template class Genten::TensorReader<SPACE>; \
  template class Genten::TensorWriter<SPACE>;

GENTEN_INST(INST_MACRO)
