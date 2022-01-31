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

#include "Genten_DistSpTensor.hpp"
#include "Genten_IOtext.hpp"

#include <algorithm>
#include <cassert>
#include <exception>
#include <fstream>
#include <random>
#include <sstream>

namespace Genten {
namespace detail {

namespace {
// This function takes a ModeLength and returns ModeLength/ProcsInMode number
// of divisions such that all modes are approximately the same size if
// ModeLength % ProcsInMode != 0 then the size of the first modes n modes will
// each be 1 larger than the remaining modes
auto UOPRSingleDimension(int ModeLength, int ProcsInMode) {
  small_vector<int> Range{0}; // Always start at 0

  const auto FibersPerBlock = ModeLength / ProcsInMode;
  auto Remainder = ModeLength % ProcsInMode;

  // We ended up with more processors than rows in the fiber :O Just return all
  // fibers in the same block. It seems easier to handle this here than to try
  // to make the while loop logic do something smart
  if (FibersPerBlock == 0) {
    Range.push_back(ModeLength);
  }

  // Make blocks
  while (Range.back() < ModeLength) {
    const auto back = Range.back();
    // This branch makes our blocks 1 bigger to eat the Remainder fibers
    if (Remainder > 0) {
      Range.push_back(back + FibersPerBlock + 1);
      --Remainder;
    } else {
      Range.push_back(back + FibersPerBlock);
    }
  }

  // Sanity check that we ended with the correct number of blocks and fibers
  assert(Range.size() == ProcsInMode + 1);
  assert(Range.back() == ModeLength);
  return Range;
}

// Blocking for uniform one per rank
auto generateUOPRBlocking(TensorInfo const &Ti,
                          small_vector<int> const &PmapGrid) {
  std::vector<small_vector<int>> blocking;
  auto const &dim_sizes = Ti.dim_sizes;
  const auto ndims = dim_sizes.size();
  blocking.reserve(ndims);

  for (auto i = 0; i < ndims; ++i) {
    blocking.emplace_back(UOPRSingleDimension(dim_sizes[i], PmapGrid[i]));
  }

  return blocking;
}

// Could optimize this to return the const& std::string in the file_name_opt,
// but I don't want to have to worry about lifetime issues.
auto getTensorFile(boost::optional<std::string> const &file_name_opt) {
  if (!file_name_opt.has_value()) {
    throw std::invalid_argument(
        "No filename in the input was found with the path tensor.file");
  }

  return file_name_opt.value();
}

} // namespace

TensorInfo
readTensorHeader(boost::optional<std::string> const &tensor_file_name) {
  auto file_name = getTensorFile(tensor_file_name);

  std::ifstream tensor_file(file_name);
  return read_sptensor_header(tensor_file);
}

namespace {

struct TensorDump {

  // This struct is dangerous since it requires the programer to do bounds
  // checking, but we are using it at the moment to avoid UB when calling
  // MPI_Send and to also let us just send Bytes over the wire instead of
  // having to serialize some more complicated stucture. 
  struct Datatype {
    // Technically calling MPI_Send on unitialized data is UB and there is
    // not an easy way to constexpr write -1 into a std::array.
    int coo[12] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    double val;
  };

  std::vector<Datatype> data;
};

// Obviously super broken on anything that isn't Host
TensorDump readTensor(std::ifstream &ifs) {
  auto index_base = 1; // Needs to come from ptree
  Genten::Sptensor X;
  import_sptensor(ifs, X, index_base, true /*verbose*/);

  TensorDump dump;
  dump.data.reserve(X.nnz());
  assert(X.ndims() <= 12);

  for (auto i = 0ll; i < X.nnz(); ++i) {
    TensorDump::Datatype dt;
    for (auto j = 0; j < X.ndims(); ++j) {
      dt.coo[j] = X.subscript(i, j);
    }

    dt.val = X.value(i);
    dump.data.emplace_back(std::move(dt));
  }

  return dump;
}

} // namespace

void tensorPlayGround(boost::optional<std::string> const &tensor_file_name,
                      TensorInfo const &Ti, ProcessorMap const &pmap,
                      std::vector<small_vector<int>> const &blocking) {
  small_vector<int> who_gets_what =
      UOPRSingleDimension(Ti.nnz, pmap.gridSize());

  TensorDump dump;
  // MPI DATA TYPES ARE ANNOYTING
  // Example here http://mpi.deino.net/mpi_functions/MPI_Type_create_struct.html
  // MPI_Datatype COOVal;
  // MPI_Datatype type[] = {MPI_INT, MPI_DOUBLE};
  // int type_num[] = {5, 1};
  // MPI_Aint displacement[2];
  //
  // DON'T DO THIS FOR NOW BECAUSE DISPLACEMENTS ARE COMPLICATED
  //
  // Because Datatype is memcpyable let's just send bytes

  if (pmap.gridRank() == 0) {
    std::cout << "who gets what: ";
    for (auto i : who_gets_what) {
      std::cout << i << " ";
    }

    std::cout << std::endl;
    auto file_name = getTensorFile(tensor_file_name);
    std::ifstream tensor_file(file_name);

    dump = readTensor(tensor_file);
    std::cout << "Dump Size: " << dump.data.size() << std::endl;

    std::vector<MPI_Request> requests(pmap.gridSize() - 1);
    std::vector<MPI_Status> statuses(pmap.gridSize() - 1);
    for (auto i = 1; i < pmap.gridSize(); ++i) {
      // Size to sent to rank i
      const auto nelements = who_gets_what[i] - who_gets_what[i - 1];
      constexpr auto dt_size = sizeof(TensorDump::Datatype);
      const auto nbytes = nelements * dt_size;

      const auto index_of_first_element = who_gets_what[i - 1];
      std::cout << "Trying to send " << nbytes << " bytes to rank " << i
                << std::endl;
      MPI_Isend(dump.data.data() + index_of_first_element, nbytes, MPI_BYTE, i,
                i, pmap.gridComm(), &requests[i - 1]);
    }
    MPI_Waitall(requests.size(), requests.data(), statuses.data());
    auto begin = dump.data.begin();
    std::advance(begin, who_gets_what[1]); // wgw[0] == 0 always
    dump.data.erase(begin, dump.data.end());
    dump.data.shrink_to_fit(); // Yay now I only have rank 0 data
  }

  if (pmap.gridRank() != 0) {
    const auto rank = pmap.gridRank();
    const auto nelements = who_gets_what[rank] - who_gets_what[rank - 1];
    dump.data.resize(nelements);
    constexpr auto dt_size = sizeof(TensorDump::Datatype);
    const auto nbytes = nelements * dt_size;
    MPI_Recv(dump.data.data(), nbytes, MPI_BYTE, 0, rank, pmap.gridComm(),
             nullptr);
  }

  if (pmap.gridRank() == pmap.gridSize() - 1) {
    std::cout << "Value: " << dump.data[0].val << "\n\tat: ";
    for (auto i = 0; i < Ti.dim_sizes.size(); ++i) {
      std::cout << dump.data[0].coo[i] << " ";
    }
    std::cout << std::endl;
  }

  MPI_Barrier(pmap.gridComm());

  auto ndims = Ti.dim_sizes.size();
  // Let's do something hacky to ship data to everyone else.
  std::vector<TensorDump> others(pmap.gridSize());
  for (auto const &dt : dump.data) {
    small_vector<int> coo(dt.coo, dt.coo + ndims); 
    auto owner_rank = rankInGridThatOwns(coo, pmap, blocking);
    others[owner_rank].data.push_back(dt);
  }

  small_vector<int> amount_to_write(pmap.gridSize(), 0);
  small_vector<int> total_amount_writen(pmap.gridSize(), 0);
  for (auto i = 0; i < pmap.gridSize(); ++i) {
    amount_to_write[i] = others[i].data.size();
  }

  MPI_Allreduce(amount_to_write.data(), total_amount_writen.data(),
                pmap.gridSize(), MPI_INT, MPI_SUM, pmap.gridComm());

  small_vector<int> offset_to_write_at(pmap.gridSize(), 0);
  MPI_Exscan(amount_to_write.data(), offset_to_write_at.data(), pmap.gridSize(),
             MPI_INT, MPI_SUM, pmap.gridComm());

  for (auto i = 0; i < pmap.gridSize(); ++i) {
    if (pmap.gridRank() == i) {
      std::cout << "Rank " << i << ": will" << std::endl;
      for (auto j = 0; j < pmap.gridSize(); ++j) {
        std::cout << "\t"
                  << " write from: " << offset_to_write_at[j] << " to "
                  << offset_to_write_at[j] + amount_to_write[j] << " on rank "
                  << j << std::endl;
      }
    }
    MPI_Barrier(pmap.gridComm());
  }

  // HERE IS THE FUN RMA PART!!!!!! Okay maybe not fun
  // MPI DATA TYPES ARE ANNOYING SO WE ARE GONNA JUST RMA BYTES INSTEAD
  // struct Datatype { // Only works for lbnl (order 5) right now
  //   int coo[5];
  //   double val;
  // };
  TensorDump::Datatype *dt;
  MPI_Win window;
  const auto my_rank = pmap.gridRank();
  constexpr auto TDsize = sizeof(TensorDump::Datatype);
  MPI_Win_allocate(total_amount_writen[my_rank] * TDsize,
                   /*displacement = */ TDsize, MPI_INFO_NULL, pmap.gridComm(),
                   &dt, &window);

  // Jonathan L. told me for AllToAll Fences are probably better than locking if
  // communication don't conflict
  MPI_Win_fence(0, window);
  for (auto i = 0; i < pmap.gridSize(); ++i) {
    const auto bytes_to_write = TDsize * amount_to_write[i];
    MPI_Put(
        /* Origin ptr */ others[i].data.data(),
        /* Origin num bytes */ bytes_to_write,
        /* Datatype for put */ MPI_BYTE,
        /* Target */ i,
        /* Displacement at target */ offset_to_write_at[i], // Note not in bytes
        /* Target num bytes */ bytes_to_write,
        /* Origin data type */ MPI_BYTE, window);
  }
  MPI_Win_fence(0, window);
  for (auto i = 0; i < pmap.gridSize(); ++i) {
    if (pmap.gridRank() == i) {
      std::cout << "Rank " << i << ": first written value " << dt[0].val
                << "\n\tat " << std::flush;
      for (auto j = 0; j < 5; ++j) {
        std::cout << dt[0].coo[j] << " ";
      }
      std::cout << std::endl;
    }
    sleep(1);
    MPI_Barrier(pmap.gridComm());
  }
  MPI_Win_free(&window);
}

std::vector<small_vector<int>>
generateBlocking(TensorInfo const &Ti, small_vector<int> const &PmapGrid,
                 TensorBlockingStrategy Bs) {
  switch (Bs) {
  case TensorBlockingStrategy::Uniform_OnePerRank_BS:
    return generateUOPRBlocking(Ti, PmapGrid);
  default:
    std::stringstream ss;
    ss << "Invalid TensorBlockingStrategy of " << int(Bs)
       << " passed to Genten::detail::generateBlocking at " << __FILE__ << ":"
       << __LINE__ << "\n";
    throw std::invalid_argument(ss.str());
  }
}

TensorBlockingStrategy readBlockingStrategy(std::string name) {
  // Convert name to all lowercase
  std::transform(name.begin(), name.end(), name.begin(),
                 [](auto c) { return std::tolower(c); });

  if (name == "default" || name == "medium") {
    return TensorBlockingStrategy::Uniform_OnePerRank_BS;
  }

  std::string error = "tensor.blocking " + name +
                      " once not one of the support types.  Supported types "
                      "are default and medium";

  throw std::invalid_argument(error);
}

namespace {
int blockInThatDim(int element, small_vector<int> const &range) {
  const auto nblocks = range.size();
  assert(element < range.back()); // This would mean the element is too large
  assert(range.size() >= 2);      // Range always has at least 2 elements

  // We could binary search, which could be faster for large ranges, but I
  // suspect this is fine. Because range.back() is always 1 more than the
  // largest possible element and range.size() >= 2 we don't have to worry
  // about block_guess + 1 going past the end.
  auto block_guess = 0;
  while (element > range[block_guess + 1]) {
    ++block_guess;
  }

  return block_guess;
}
} // namespace

int rankInGridThatOwns(small_vector<int> const &COO, ProcessorMap const &pmap,
                       std::vector<small_vector<int>> const &ElementRanges) {
  const auto ndims = COO.size();
  assert(ndims == ElementRanges.size());

  small_vector<int> GridPos(ndims);
  for (auto i = 0; i < ndims; ++i) {
    GridPos[i] = blockInThatDim(COO[i], ElementRanges[i]);
  }

  int rank;
  MPI_Cart_rank(pmap.gridComm(), GridPos.data(), &rank);

  return rank;
}

} // namespace detail
} // namespace Genten
