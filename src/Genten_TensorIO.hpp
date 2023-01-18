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

#pragma once

#include <string>

#include "Genten_Sptensor.hpp"
#include "Genten_Tensor.hpp"

namespace Genten {

// Type to temporarily hold coo data for MPI distribution
struct SpDataType {
  ttb_indx coo[6] = {-1u, -1u, -1u, -1u, -1u, -1u};
  ttb_real val;
};

// Info from reading a sparse tensor header
struct SptnFileHeader {
  using nd_type = uint32_t;
  using nnz_type = uint64_t;
  using dim_type = uint64_t;
  using sub_size_type = uint64_t;
  using float_size_type = uint32_t;

  SptnFileHeader() = default;
  SptnFileHeader(const Sptensor& X, const float_size_type float_size);

  nd_type ndims = 0;
  float_size_type float_bits = 0;
  small_vector<dim_type> dim_lengths;
  small_vector<sub_size_type> dim_bits;
  nnz_type nnz = 0;
  std::uint64_t data_starting_byte = 0;

  std::uint64_t bytesInDataLine() const;
  std::uint64_t indByteOffset(int ind) const;
  std::uint64_t dataByteOffset() const;
  std::uint64_t totalBytesToRead() const;

  small_vector<std::uint64_t> getOffsetRanges(int nranks) const;

  std::pair<std::uint64_t, std::uint64_t> getLocalOffsetRange(int rank,
                                                              int nranks) const;
  std::vector<ttb_indx> getGlobalDims() const;
  ttb_indx getGlobalNnz() const;

  void readBinary(std::istream& in);
  void writeBinary(std::ostream& out);
};

// Info from reading a dense tensor header
struct DntnFileHeader {
  using nd_type = uint32_t;
  using nnz_type = uint64_t;
  using dim_type = uint64_t;
  using float_size_type = uint32_t;

  DntnFileHeader() = default;
  DntnFileHeader(const Tensor& X, const float_size_type float_size);

  nd_type ndims = 0;
  float_size_type float_bits = 0;
  small_vector<dim_type> dim_lengths;
  nnz_type nnz = 0;
  std::uint64_t data_starting_byte = 0;

  std::uint64_t bytesInDataLine() const { return float_bits / 8; }
  std::uint64_t totalBytesToRead() const { return bytesInDataLine() * nnz; }

  small_vector<std::uint64_t> getOffsetRanges(int nranks) const;

  std::pair<std::uint64_t, std::uint64_t> getLocalOffsetRange(int rank,
                                                              int nranks) const;
  std::vector<ttb_indx> getGlobalDims() const;
  ttb_indx getGlobalNnz() const;

  void readBinary(std::istream& in);
  void writeBinary(std::ostream& out);
};

template <typename ExecSpace>
class TensorReader {
public:
  TensorReader(const std::string& filename,
               const ttb_indx index_base = 0,
               const bool compressed = false);

  void read();

#ifdef HAVE_DIST
  std::vector<SpDataType>
  parallelReadBinarySparse(std::vector<ttb_indx>& global_dims,
                           ttb_indx& nnz) const;

  std::vector<ttb_real>
  parallelReadBinaryDense(std::vector<ttb_indx>& global_dims,
                          ttb_indx& nnz,
                          ttb_indx& offset) const;
#endif

  bool isSparse() const { return is_sparse; }
  bool isDense() const { return is_dense; }
  bool isBinary() const { return is_binary; }
  bool isText() const { return is_text; }
  SptensorT<ExecSpace> getSparseTensor() { return X_sparse; }
  TensorT<ExecSpace> getDenseTensor() { return X_dense; }

private:
  std::string filename;
  ttb_indx index_base;
  bool compressed;

  bool is_sparse;
  bool is_dense;
  bool is_binary;
  bool is_text;

  SptensorT<ExecSpace> X_sparse;
  TensorT<ExecSpace> X_dense;

  void queryFile();
};

template <typename ExecSpace>
class TensorWriter {
public:
  TensorWriter(const std::string& filename,
               const bool compressed = false);

  void writeBinary(const SptensorT<ExecSpace>& X) const;
  void writeBinary(const TensorT<ExecSpace>& X) const;

  void writeText(const SptensorT<ExecSpace>& X) const;
  void writeText(const TensorT<ExecSpace>& X) const;
private:
  std::string filename;
  bool compressed;
};

}
