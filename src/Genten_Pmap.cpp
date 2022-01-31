//@header
// ************************************************************************
//     genten: software for generalized tensor decompositions
//     by sandia national laboratories
//
// sandia national laboratories is a multimission laboratory managed
// and operated by national technology and engineering solutions of sandia,
// llc, a wholly owned subsidiary of honeywell international, inc., for the
// u.s. department of energy's national nuclear security administration under
// contract de-na0003525.
//
// copyright 2017 national technology & engineering solutions of sandia, llc
// (ntess). under the terms of contract de-na0003525 with ntess, the u.s.
// government retains certain rights in this software.
//
// redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// this software is provided by the copyright holders and contributors
// "as is" and any express or implied warranties, including, but not
// limited to, the implied warranties of merchantability and fitness for
// a particular purpose are disclaimed. in no event shall the copyright
// holder or contributors be liable for any direct, indirect, incidental,
// special, exemplary, or consequential damages (including, but not
// limited to, procurement of substitute goods or services; loss of use,
// data, or profits; or business interruption) however caused and on any
// theory of liability, whether in contract, strict liability, or tort
// (including negligence or otherwise) arising in any way out of the use
// of this software, even if advised of the possibility of such damage.
// ************************************************************************
//@header

#include "Genten_Pmap.hpp"
#include "Genten_DistContext.hpp"
#include "Genten_IOtext.hpp"

#include <boost/serialization/vector.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace Genten {

namespace {
// Silly function to compute divisors
auto divisors(int input) {
  small_vector<int> divisors(1, input);
  int sroot = std::sqrt(input);
  for (auto i = 1; i <= sroot; ++i) {
    if (input % i == 0) {
      divisors.push_back(i);
      if (i > 1 && i != sroot) {
        divisors.push_back(input / i);
      }
    }
  }

  std::sort(divisors.begin(), divisors.end());
  return divisors;
}

// Goal is to count the total storage of the factors for the given grid Storage
// of each factor is the size of the factor matrix times the number of
// processes in the grid that are not in our fiber.
//
// clang-format off
// For example given a grid [2, 3, 5, 7] factor matrices would be distributed over: 
// F0: [_, 3, 5, 7] = 105 of the 210 processes
// F1: [2, _, 5, 7] = 70 of the 210 processes
// F2: [2, 3, _, 7] = 42 of the 210 processes
// F3: [2, 3, 5, _] = 30 of the 210 processes
// clang-format on
//
// Then to compute the total storage you need to multiplie the size of each
// factor matrix times the number of processes it is distributed over.
//
// To keep this code from needing to know about the rank of the factors we will
// return the result for rank 1 factors. The calling code can simply scale this
// result by the rank to figure out the total number of elements
auto nelementsForRank1Factors(small_vector<int> const &grid,
                              std::vector<std::uint32_t> const &tensor_dims) {
  auto nprocs =
      std::accumulate(grid.begin(), grid.end(), 1ll, std::multiplies<>{});

  const auto ndims = grid.size();
  int64_t nelements = 0;
  for (auto i = 0; i < ndims; ++i) {
    const auto replicated_procs = nprocs / grid[i];
    nelements += replicated_procs * tensor_dims[i];
  }

  return nelements;
}

// This function writes the grid with that leads to the minimal storage
// required for the factor matrices
auto recurseMinSpaceGrid(int nprocs, small_vector<int> &grid,
                         std::vector<std::uint32_t> const &tensor_dims,
                         int dims_remaining) {
  assert(dims_remaining >= 1);

  // The last index has no freedom just set it and return
  if (dims_remaining == 1) {
    grid.back() = nprocs;
    return;
  }

  // Current index tells us which position we are in
  const auto current_index = grid.size() - dims_remaining;

  // Make copy for testing on so that we only ever write to grid when we've
  // found a better option
  auto test = grid;
  auto min_storage = std::numeric_limits<int64_t>::max();

  for (auto d : divisors(nprocs)) {
    test[current_index] = d;
    const auto remaining_procs = nprocs / d;
    recurseMinSpaceGrid(remaining_procs, test, tensor_dims, dims_remaining - 1);

    auto test_storage = nelementsForRank1Factors(test, tensor_dims);
    if (test_storage < min_storage) {
      min_storage = test_storage;
      grid = test;
    }
  }
}

auto minFactorSpaceGrid(int nprocs,
                        std::vector<std::uint32_t> const &tensor_dims) {
  const auto ndims = tensor_dims.size();
  auto grid = small_vector<int>(ndims);
  if (DistContext::rank() == 0) {
    recurseMinSpaceGrid(nprocs, grid, tensor_dims, ndims);
  }
  DistContext::Bcast(grid, 0);
  return grid;
}

small_vector<int> CartGrid(int nprocs,
                           std::vector<std::uint32_t> const &tensor_dims);

small_vector<int> singleDimUniformBlocking(int ModeLength, int ProcsInMode) {
  small_vector<int> Range{0};
  const auto FibersPerBlock = ModeLength / ProcsInMode;
  auto Remainder = ModeLength % ProcsInMode;

  // We ended up with more processors than rows in the fiber :O Just return all
  // fibers in the same block. It seems easier to handle this here than to try
  // to make the while loop logic do something smart
  if (FibersPerBlock == 0) {
    Range.push_back(ModeLength);
  }

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

std::vector<small_vector<int>>
generateUniformBlocking(std::vector<std::uint32_t> const &ModeLengths,
                        small_vector<int> const &ProcGridSizes) {
  const auto Ndims = ModeLengths.size();
  std::vector<small_vector<int>> blocking;
  blocking.reserve(Ndims);

  for (auto i = 0; i < Ndims; ++i) {
    blocking.emplace_back(
        singleDimUniformBlocking(ModeLengths[i], ProcGridSizes[i]));
  }

  return blocking;
}

small_vector<double>
testPmapGrid(ProcessorMap const &pmap, int factor_rank,
             std::vector<std::uint32_t> const &tensor_dims) {
  small_vector<std::vector<double>> test_factors;
  auto blocking = generateUniformBlocking(tensor_dims, pmap.gridDims());

  const auto ndims = tensor_dims.size();
  for (auto i = 0; i < ndims; ++i) {
    auto const &blocking_dim = blocking[i];
    const int grid_cord = pmap.gridCoord(i);
    const int nrows_dim = blocking_dim[grid_cord + 1] - blocking_dim[grid_cord];

    test_factors.emplace_back(nrows_dim * factor_rank, 0.0);
  }

  small_vector<double> times(ndims, 0.0);
  auto run_allreduce = [&] {
    for (auto i = 0; i < ndims; ++i) {
      if (pmap.subCommSize(i) == 1) { // Don't Allreduce when not replicated
        continue;
      }

      auto t0 = MPI_Wtime();
      MPI_Allreduce(MPI_IN_PLACE, &test_factors[i][0], test_factors[i].size(),
                    MPI_DOUBLE, MPI_SUM, pmap.subComm(i));

      times[i] += MPI_Wtime() - t0;
    }
  };

  // Time 5 iterations of AllReduce to try and get something reasonable
  for (auto i = 0; i < 20; ++i) {
    run_allreduce();
  }

  MPI_Allreduce(MPI_IN_PLACE, &times[0], ndims, MPI_DOUBLE, MPI_SUM,
                pmap.gridComm());

  return times;
}

void updateGuessGrid(small_vector<int> &grid,
                     small_vector<double> const &scores) {
  const auto ndim = scores.size();
  auto min_max = std::minmax_element(scores.begin(), scores.end());
  auto min_idx = std::distance(scores.begin(), min_max.first);
  auto max_idx = std::distance(scores.begin(), min_max.second);

  auto divisors_min = divisors(grid[min_idx]);
  if (divisors_min.size() >= 2) {
    auto scale = divisors_min[1];
    grid[max_idx] *= scale;
    grid[min_idx] /= scale;
  }
}

auto minAllReduceComm(int nprocs, int factor_rank,
                      std::vector<std::uint32_t> const &tensor_dims) {
  auto guess_grid = minFactorSpaceGrid(nprocs, tensor_dims);
  small_vector<int> chosen_grid(guess_grid.size());
  bool decided = false;
  auto score = std::numeric_limits<double>::max();
  while (!decided) {
    if (DistContext::rank() == 0 && DistContext::isDebug()) {
      std::cout << "Testing processor grid: ";
      for (auto i : guess_grid) {
        std::cout << i << " ";
      }
      std::cout << std::endl;
    }
    DistContext::Barrier();
    ProcessorMap pmap({}, tensor_dims, guess_grid);
    auto guess_scores = testPmapGrid(pmap, factor_rank, tensor_dims);
    auto guess_score =
        std::accumulate(guess_scores.begin(), guess_scores.end(), 0.0);
    if (DistContext::rank() == 0 && DistContext::isDebug()) {
      std::cout << "\tGrid scored: " << guess_score << ", ";
      for (auto s : guess_scores) {
        std::cout << s << " ";
      }
      std::cout << std::endl;
    }
    DistContext::Barrier();
    if (guess_score < score) {
      chosen_grid = guess_grid;
      score = guess_score;
      updateGuessGrid(guess_grid, guess_scores);
    } else {
      decided = true;
    }
  }

  return chosen_grid;
}

small_vector<int> CartGrid(int nprocs,
                           std::vector<std::uint32_t> const &tensor_dims) {
  return minFactorSpaceGrid(nprocs, tensor_dims);
}
} // namespace

ProcessorMap::ProcessorMap(ptree const &input_tree,
                           std::vector<std::uint32_t> const &tensor_dims,
                           small_vector<int> const &predetermined_grid)
    : dimension_sizes_(predetermined_grid),
      pmap_tree_(input_tree.get_child("pmap", ptree{})) {
  const auto ndims = dimension_sizes_.size();

  // I don't think we need to be periodic
  small_vector<int> periodic(ndims, 0);
  bool reorder = true; // Let MPI be smart I guess
  MPI_Cart_create(DistContext::commWorld(), ndims, dimension_sizes_.data(),
                  periodic.data(), reorder, &cart_comm_);

  MPI_Comm_size(cart_comm_, &grid_nprocs_);
  MPI_Comm_rank(cart_comm_, &grid_rank_);
  coord_.resize(ndims);
  MPI_Cart_coords(cart_comm_, grid_rank_, ndims, coord_.data());

  small_vector<int> dim_filter(ndims, 1);
  sub_maps_.resize(ndims);
  sub_grid_rank_.resize(ndims);
  sub_comm_sizes_.resize(ndims);

  // Get information for the MPI Subgrid for each Dimension
  for (auto i = 0; i < ndims; ++i) {
    dim_filter[i] = 0; // Get all dims except this one
    MPI_Cart_sub(cart_comm_, dim_filter.data(), &sub_maps_[i]);
    dim_filter[i] = 1; // Reset the dim_filter

    MPI_Comm_rank(sub_maps_[i], &sub_grid_rank_[i]);
    MPI_Comm_size(sub_maps_[i], &sub_comm_sizes_[i]);
  }
}

ProcessorMap::ProcessorMap(ptree const &input_tree,
                           std::vector<std::uint32_t> const &tensor_dims)
    : ProcessorMap(input_tree, tensor_dims,
                   CartGrid(DistContext::nranks(), tensor_dims)) {}

void ProcessorMap::gridBarrier() const {
  if (grid_nprocs_ > 1) {
    MPI_Barrier(cart_comm_);
  }
}

ProcessorMap::~ProcessorMap() {
  if (DistContext::initialized()) {
    for (auto &comm : sub_maps_) {
      if (comm != MPI_COMM_NULL) {
        MPI_Comm_free(&comm);
      }
    }
    if (cart_comm_ != MPI_COMM_NULL) {
      MPI_Comm_free(&cart_comm_);
    }
  }
}
} // namespace Genten
