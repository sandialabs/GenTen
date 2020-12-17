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

#include "Genten_DistSpSystem.hpp"
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
  auto const &dim_sizes = Ti.sizes;
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
  auto header_info = read_sptensor_header(tensor_file);
  TensorInfo Ti;
  Ti.nnz = header_info.nnz;
  Ti.sizes.resize(header_info.dim_sizes.size());
  std::copy(header_info.dim_sizes.begin(), header_info.dim_sizes.end(),
            Ti.sizes.begin());
  return Ti;
}

void readTensorToRank0(boost::optional<std::string> const &tensor_file_name,
                       TensorInfo const &Ti,
                       std::vector<small_vector<int>> const &blocking) {
  auto file_name = getTensorFile(tensor_file_name);
  std::ifstream tensor_file(file_name);

  // Need to get this info from ptree after testing, using 1 for now since that
  // is what lbnl is
  auto index_base = 1;
  Genten::Sptensor X;
  import_sptensor(tensor_file, X, index_base, true /*verbose*/);

  auto bits = std::mt19937(std::random_device{}());
  auto dist = std::uniform_int_distribution<int>(0, X.nnz());

  for (auto i = 0; i < 10; ++i) {
    auto random_cord = X.getSubscripts(dist(bits));
    small_vector<int> copy(X.ndims());
    for (auto i = 0; i < X.ndims(); ++i) {
      copy[i] = random_cord[i]; // Assume OpenMP
    }
    rankInGridThatOwns(copy, blocking);
  }
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

int rankInGridThatOwns(small_vector<int> const &COO,
                       std::vector<small_vector<int>> const &ElementRanges) {
  const auto ndims = COO.size();
  assert(ndims == ElementRanges.size());

  small_vector<int> GridPos(ndims);
  for (auto i = 0; i < ndims; ++i) {
    GridPos[i] = blockInThatDim(COO[i], ElementRanges[i]);
  }

  std::cout << "Element COO: ";
  for (auto i = 0; i < ndims; ++i) {
    std::cout << COO[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Range:\n";
  for (auto i = 0; i < ndims; ++i) {
    std::cout << "\t" << i << ": ";
    for (auto pos : ElementRanges[i]) {
      std::cout << pos << " ";
    }
    std::cout << ": Position in grid: " << GridPos[i] << "\n";
  }
  std::cout << std::endl;

  return 0;
}

} // namespace detail
} // namespace Genten
