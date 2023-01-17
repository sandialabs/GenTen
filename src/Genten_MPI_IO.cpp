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
#include <memory>
#include <limits>
#include <stdexcept>

#include "Genten_MPI_IO.hpp"

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

std::vector<ttb_indx>
SptnFileHeader::getGlobalDims() const
{
  std::vector<ttb_indx> dims(ndims);
  std::copy(dim_lengths.begin(), dim_lengths.end(), dims.begin());
  return dims;
}

ttb_indx
SptnFileHeader::getGlobalNnz() const { return nnz; }

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

std::vector<ttb_indx>
DntnFileHeader::getGlobalDims() const
{
  std::vector<ttb_indx> dims(ndims);
  std::copy(dim_lengths.begin(), dim_lengths.end(), dims.begin());
  return dims;
}

ttb_indx
DntnFileHeader::getGlobalNnz() const { return nnz; }

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

SptnFileHeader readSparseHeader(MPI_Comm comm, MPI_File fh) {
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

DntnFileHeader readDenseHeader(MPI_Comm comm, MPI_File fh) {
  DntnFileHeader header;
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
    if (tag != "Dntn") {
      std::cout << "Not a binary dense tensor file" << std::endl;
      MPI_Abort(comm, MPI_ERR_UNKNOWN);
    }

    check_read(&header.ndims, sizeof header.ndims);
    const auto ndims = header.ndims;

    check_read(&header.float_bits, sizeof header.float_bits);

    header.dim_lengths.resize(header.ndims);
    check_read(&header.dim_lengths[0], ndims * sizeof(std::uint64_t));

    check_read(&header.nnz, sizeof header.nnz);

    MPI_Bcast(&header.ndims, 1, MPI_UNSIGNED, 0, comm);
    MPI_Bcast(&header.float_bits, 1, MPI_UNSIGNED, 0, comm);
    MPI_Bcast(&header.dim_lengths[0], ndims, MPI_UNSIGNED_LONG_LONG, 0, comm);
    MPI_Bcast(&header.nnz, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
    MPI_Bcast(&header.data_starting_byte, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
  } else {
    MPI_Bcast(&header.ndims, 1, MPI_UNSIGNED, 0, comm);

    const auto ndims = header.ndims;
    header.dim_lengths.resize(ndims);

    MPI_Bcast(&header.float_bits, 1, MPI_UNSIGNED, 0, comm);
    MPI_Bcast(&header.dim_lengths[0], ndims, MPI_UNSIGNED_LONG_LONG, 0, comm);
    MPI_Bcast(&header.nnz, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
    MPI_Bcast(&header.data_starting_byte, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
  }

  return header;
}

namespace {
SpDataType readSparseElement(SptnFileHeader const &h,
                             unsigned char const *const data_ptr) {
  SpDataType data;
  const auto ndims = h.ndims;
  auto const &bits = h.dim_bits;

  auto read_index = [](auto byte_ptr, int bits_for_index) -> std::uint64_t {
    switch (bits_for_index) {
    case 16:
      return *reinterpret_cast<std::uint16_t const *>(byte_ptr);
    case 32:
      return *reinterpret_cast<std::uint32_t const *>(byte_ptr);
    default:
      return *reinterpret_cast<std::uint64_t const *>(byte_ptr);
    }
  };

  for (auto i = 0; i < ndims; ++i) {
    data.coo[i] = read_index(data_ptr + h.indByteOffset(i), bits[i]);
  }

  if (h.float_bits == 32) {
    data.val =
        *reinterpret_cast<float const *>(data_ptr + h.dataByteOffset());
  } else if (h.float_bits == 64) {
    data.val =
        *reinterpret_cast<double const *>(data_ptr + h.dataByteOffset());
  } else {
    throw std::runtime_error("Currently the floating point data must have a "
                             "precision of 32 or 64 bits.");
  }

  return data;
}

ttb_real readDenseElement(DntnFileHeader const &h,
                          unsigned char const *const data_ptr) {
  ttb_real data;

  if (h.float_bits == 32) {
    data = *reinterpret_cast<float const *>(data_ptr);
  } else if (h.float_bits == 64) {
    data = *reinterpret_cast<double const *>(data_ptr);
  } else {
    throw std::runtime_error("Currently the floating point data must have a "
                             "precision of 32 or 64 bits.");
  }

  return data;
}
} // namespace

std::vector<SpDataType>
parallelReadElements(MPI_Comm comm, MPI_File fh, SptnFileHeader const &h) {
  int rank, nprocs;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);

  //const auto ndims = h.ndims;
  const auto bytes_per_element = h.bytesInDataLine();
  const auto local_range = h.getLocalOffsetRange(rank, nprocs);
  const auto nlocal_bytes = local_range.second - local_range.first;
  const auto nlocal_elements = nlocal_bytes / bytes_per_element;

  if (nlocal_elements > std::numeric_limits<int>::max()) {
    std::cout << "Rank: " << rank << " trying to read: " << nlocal_elements
              << ", but MPI can't read more than: "
              << std::numeric_limits<int>::max() << "elements in one shot."
              << std::endl;
    MPI_Abort(comm, MPI_ERR_UNKNOWN);
  }

  MPI_Datatype element_type;
  MPI_Type_contiguous(bytes_per_element, MPI_BYTE, &element_type);
  MPI_Type_commit(&element_type);

  std::vector<SpDataType> out;
  out.reserve(nlocal_elements);

  // Get the local data from the file
  MPI_Barrier(comm);
  auto t0 = MPI_Wtime();
  auto byte_array = std::make_unique<unsigned char[]>(nlocal_bytes);
  int error =
      MPI_File_read_at_all(fh, local_range.first, byte_array.get(),
                           nlocal_elements, element_type, MPI_STATUSES_IGNORE);
  if (error != MPI_SUCCESS) {
    MPI_Abort(comm, error);
  }
  MPI_Barrier(comm);
  auto t1 = MPI_Wtime();
  if (0 && rank == 0) { // Turning this off for now
    std::cout << "\tTime in MPI_File_read_at_all: " << t1 - t0 << "s"
              << std::endl;
  }

  MPI_Type_free(&element_type);

  // Fill the vector
  for (auto i = 0; i < nlocal_elements; ++i) {
    auto curr = byte_array.get() + i * bytes_per_element;
    out.push_back(readSparseElement(h, curr));
  }

  return out;
}

std::vector<ttb_real>
parallelReadElements(MPI_Comm comm, MPI_File fh, DntnFileHeader const &h) {
  int rank, nprocs;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);

  //const auto ndims = h.ndims;
  const auto bytes_per_element = h.bytesInDataLine();
  const auto local_range = h.getLocalOffsetRange(rank, nprocs);
  const auto nlocal_bytes = local_range.second - local_range.first;
  const auto nlocal_elements = nlocal_bytes / bytes_per_element;

  if (nlocal_elements > std::numeric_limits<int>::max()) {
    std::cout << "Rank: " << rank << " trying to read: " << nlocal_elements
              << ", but MPI can't read more than: "
              << std::numeric_limits<int>::max() << "elements in one shot."
              << std::endl;
    MPI_Abort(comm, MPI_ERR_UNKNOWN);
  }

  MPI_Datatype element_type;
  MPI_Type_contiguous(bytes_per_element, MPI_BYTE, &element_type);
  MPI_Type_commit(&element_type);

  std::vector<ttb_real> out;
  out.reserve(nlocal_elements);

  // Get the local data from the file
  MPI_Barrier(comm);
  auto t0 = MPI_Wtime();
  auto byte_array = std::make_unique<unsigned char[]>(nlocal_bytes);
  int error =
      MPI_File_read_at_all(fh, local_range.first, byte_array.get(),
                           nlocal_elements, element_type, MPI_STATUSES_IGNORE);
  if (error != MPI_SUCCESS) {
    MPI_Abort(comm, error);
  }
  MPI_Barrier(comm);
  auto t1 = MPI_Wtime();
  if (0 && rank == 0) { // Turning this off for now
    std::cout << "\tTime in MPI_File_read_at_all: " << t1 - t0 << "s"
              << std::endl;
  }

  MPI_Type_free(&element_type);

  // Fill the vector
  for (auto i = 0; i < nlocal_elements; ++i) {
    auto curr = byte_array.get() + i * bytes_per_element;
    out.push_back(readDenseElement(h, curr));
  }

  return out;
}

} // namespace G_MPI_IO
} // namespace Genten
