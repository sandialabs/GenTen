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
#include <Genten_MPI_IO.h>

#include <cassert>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace Genten {
namespace MPI_IO {

MPI_File openFile(MPI_Comm comm, std::string const &file_name, int access_mode,
                  MPI_Info info) {
  MPI_File fh;
  auto error = MPI_File_open(comm, file_name.c_str(), access_mode, info, &fh);
  if (error != 0) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
      std::cout << "There was an error opening file: " << file_name
                << " terminating job." << std::endl;
    }
    MPI_Barrier(comm);
    MPI_Abort(comm, error);
  }
  return fh;
}

std::uint64_t SptnFileHeader::bytesInDataLine() const {
  return std::accumulate(dim_bits.begin(), dim_bits.end(), float_bits) / 8;
}

std::uint64_t SptnFileHeader::dataByteOffset() const {
  return std::accumulate(dim_bits.begin(), dim_bits.end(), 0) / 8;
}

std::uint64_t SptnFileHeader::indByteOffset(int ind) const {
  if (ind >= ndims) {
    throw std::out_of_range(
        "Called indByteOffset with index that was out of range\n");
  }
  auto it = dim_bits.begin();
  std::advance(it, ind);
  return std::accumulate(dim_bits.begin(), it, 0) / 8;
}

std::uint64_t SptnFileHeader::totalBytesToRead() const {
  return bytesInDataLine() * nnz;
}

std::vector<std::uint64_t> SptnFileHeader::getOffsetRanges(int nranks) const {
  const auto nper_rank = nnz / nranks;
  assert(nper_rank != 0);

  std::vector<std::uint64_t> out;
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
  return {range[rank], range[rank+1]};
};

std::ostream &operator<<(std::ostream &os, SptnFileHeader const &h) {
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

SptnFileHeader readHeader(MPI_Comm comm, MPI_File fh) {
  SptnFileHeader header;
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == 0) {
    auto check_read = [&](void *buffer, int num_bytes) {
      int error =
          MPI_File_read(fh, buffer, num_bytes, MPI_BYTE, MPI_STATUS_IGNORE);

      if (error != MPI_SUCCESS) {
        MPI_Abort(comm, error);
      }

      // A bit hacky to put this here but what ever
      header.data_starting_byte += num_bytes;
    };

    std::string tag = "xxxx";
    check_read(&tag[0], 4);
    if (tag != "sptn") {
      std::cout << "Not a binary sparse tensor file" << std::endl;
      MPI_Abort(comm, MPI_ERR_UNKNOWN);
    }

    check_read(&header.ndims, sizeof header.ndims);
    const auto ndims = header.ndims;

    check_read(&header.float_bits, sizeof header.float_bits);

    header.dim_lengths.resize(header.ndims);
    check_read(&header.dim_lengths[0], ndims * sizeof(std::uint64_t));

    header.dim_bits.resize(header.ndims);
    check_read(&header.dim_bits[0], ndims * sizeof(std::uint64_t));

    check_read(&header.nnz, sizeof header.nnz);

    MPI_Bcast(&header.ndims, 1, MPI_UNSIGNED, 0, comm);
    MPI_Bcast(&header.float_bits, 1, MPI_UNSIGNED, 0, comm);
    MPI_Bcast(&header.dim_lengths[0], ndims, MPI_UNSIGNED_LONG_LONG, 0, comm);
    MPI_Bcast(&header.dim_bits[0], ndims, MPI_UNSIGNED_LONG_LONG, 0, comm);
    MPI_Bcast(&header.nnz, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
    MPI_Bcast(&header.data_starting_byte, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
  } else {
    MPI_Bcast(&header.ndims, 1, MPI_UNSIGNED, 0, comm);

    const auto ndims = header.ndims;
    header.dim_lengths.resize(ndims);
    header.dim_bits.resize(ndims);

    MPI_Bcast(&header.float_bits, 1, MPI_UNSIGNED, 0, comm);
    MPI_Bcast(&header.dim_lengths[0], ndims, MPI_UNSIGNED_LONG_LONG, 0, comm);
    MPI_Bcast(&header.dim_bits[0], ndims, MPI_UNSIGNED_LONG_LONG, 0, comm);
    MPI_Bcast(&header.nnz, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
    MPI_Bcast(&header.data_starting_byte, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
  }

  return header;
}

} // namespace MPI_IO
} // namespace Genten
