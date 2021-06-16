#pragma once
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

#include <mpi.h>

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace Genten {
namespace MPI_IO {
MPI_File openFile(MPI_Comm comm, std::string const &file_name,
                  int access_mode = MPI_MODE_RDONLY,
                  MPI_Info info = MPI_INFO_NULL);

struct SptnFileHeader {
  std::uint32_t ndims = 0;
  std::uint32_t float_bits = 0;
  std::vector<std::uint64_t> dim_lengths;
  std::vector<std::uint64_t> dim_bits;
  std::uint64_t nnz = 0;
  std::uint64_t data_starting_byte = 0;

  std::uint64_t bytesInDataLine() const;
  std::uint64_t indByteOffset(int ind) const;
  std::uint64_t dataByteOffset() const;
  std::uint64_t totalBytesToRead() const;

  std::vector<std::uint64_t> getOffsetRanges(int nranks) const;

  std::pair<std::uint64_t, std::uint64_t> getLocalOffsetRange(int rank,
                                                              int nranks) const;
};

std::ostream &operator<<(std::ostream &os, SptnFileHeader const &h);

SptnFileHeader readHeader(MPI_Comm comm, MPI_File fh);

} // namespace MPI_IO
} // namespace Genten
