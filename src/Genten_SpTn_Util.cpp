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
//

#include <iostream>
#include <cassert>
#include <numeric>
#include <stdexcept>

#include "Genten_SpTn_Util.hpp"

namespace Genten {
namespace G_MPI_IO {

std::uint64_t
SptnFileHeader::bytesInDataLine() const {
  return std::accumulate(dim_bits.begin(), dim_bits.end(), float_bits) / 8;
}

std::uint64_t
SptnFileHeader::dataByteOffset() const {
  return std::accumulate(dim_bits.begin(), dim_bits.end(), 0) / 8;
}

std::uint64_t
SptnFileHeader::indByteOffset(int ind) const {
  if (ind >= ndims) {
    throw std::out_of_range(
        "Called indByteOffset with index that was out of range\n");
  }
  auto it = dim_bits.begin();
  std::advance(it, ind);
  return std::accumulate(dim_bits.begin(), it, 0) / 8;
}

std::uint64_t
SptnFileHeader::totalBytesToRead() const {
  return bytesInDataLine() * nnz;
}

small_vector<std::uint64_t>
SptnFileHeader::getOffsetRanges(int nranks) const {
  const auto nper_rank = nnz / nranks;
  assert(nper_rank != 0);

  small_vector<std::uint64_t> out;
  out.reserve(nranks + 1);

  const auto line_bytes = bytesInDataLine();
  std::uint64_t starting_elem = 0;
  for (auto i = 0; i < nranks; ++i) {
    out.push_back(starting_elem * line_bytes + data_starting_byte);
    starting_elem += nper_rank;
  }
  out.push_back(nnz * line_bytes + data_starting_byte);

  return out;
}

std::pair<std::uint64_t, std::uint64_t>
SptnFileHeader::getLocalOffsetRange(int rank, int nranks) const {
  // This is overkill and I don't care
  const auto range = getOffsetRanges(nranks);
  return {range[rank], range[rank + 1]};
};

TensorInfo
SptnFileHeader::toTensorInfo() const {
  TensorInfo Ti;

  Ti.nnz = nnz;
  Ti.dim_sizes.resize(ndims);
  std::copy(dim_lengths.begin(), dim_lengths.end(), Ti.dim_sizes.begin());

  return Ti;
}

small_vector<std::uint64_t>
DntnFileHeader::getOffsetRanges(int nranks) const {
  const auto nper_rank = nnz / nranks;
  assert(nper_rank != 0);

  small_vector<std::uint64_t> out;
  out.reserve(nranks + 1);

  const auto line_bytes = bytesInDataLine();
  std::uint64_t starting_elem = 0;
  for (auto i = 0; i < nranks; ++i) {
    out.push_back(starting_elem * line_bytes + data_starting_byte);
    starting_elem += nper_rank;
  }
  out.push_back(nnz * line_bytes + data_starting_byte);

  return out;
}

std::pair<std::uint64_t, std::uint64_t>
DntnFileHeader::getLocalOffsetRange(int rank, int nranks) const {
  // This is overkill and I don't care
  const auto range = getOffsetRanges(nranks);
  return {range[rank], range[rank + 1]};
};

TensorInfo
DntnFileHeader::toTensorInfo() const {
  TensorInfo Ti;

  Ti.nnz = nnz;
  Ti.dim_sizes.resize(ndims);
  std::copy(dim_lengths.begin(), dim_lengths.end(), Ti.dim_sizes.begin());

  return Ti;
}

std::ostream
&operator<<(std::ostream &os, SptnFileHeader const &h) {
  os << "Sparse Tensor Info :\n";
  os << "\tDimensions : " << h.ndims << "\n";
  os << "\tFloat bits : " << h.float_bits << "\n";
  os << "\tSizes      : ";
  for (auto s : h.dim_lengths) {
    os << s << " ";
  }
  os << "\n";
  os << "\tIndex bits : ";
  for (auto s : h.dim_bits) {
    os << s << " ";
  }
  os << "\n";
  os << "\tNNZ        : " << h.nnz << "\n";
  os << "\tData Byte  : " << h.data_starting_byte;

  return os;
}

std::ostream
&operator<<(std::ostream &os, DntnFileHeader const &h) {
  os << "Dense Tensor Info :\n";
  os << "\tDimensions : " << h.ndims << "\n";
  os << "\tFloat bits : " << h.float_bits << "\n";
  os << "\tSizes      : ";
  for (auto s : h.dim_lengths) {
    os << s << " ";
  }
  os << "\n";
  os << "\tNNZ        : " << h.nnz << "\n";
  os << "\tData Byte  : " << h.data_starting_byte;

  return os;
}

} // namespace G_MPI_IO
} // namespace Genten
