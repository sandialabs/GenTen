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

#ifdef HAVE_DIST

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "Genten_DistContext.hpp"
#include "Genten_IOtext.hpp"

namespace Genten {

namespace {
// Silly function to compute divisors
small_vector<ttb_indx> divisors(ttb_indx input) {
  small_vector<ttb_indx> divisors(1, input);
  ttb_indx sroot = std::sqrt(input);
  for (ttb_indx i = 1; i <= sroot; ++i) {
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
ttb_indx nelementsForRank1Factors(small_vector<ttb_indx> const &grid,
                              std::vector<ttb_indx> const &tensor_dims) {
  ttb_indx nprocs =
      std::accumulate(grid.begin(), grid.end(), 1ll, std::multiplies<>{});

  const ttb_indx ndims = grid.size();
  ttb_indx nelements = 0;
  for (ttb_indx i = 0; i < ndims; ++i) {
    const ttb_indx replicated_procs = nprocs / grid[i];
    nelements += replicated_procs * tensor_dims[i];
  }

  return nelements;
}

// This function writes the grid with that leads to the minimal storage
// required for the factor matrices
void recurseMinSpaceGrid(ttb_indx nprocs, small_vector<ttb_indx> &grid,
                         std::vector<ttb_indx> const &tensor_dims,
                         ttb_indx dims_remaining) {
  assert(dims_remaining >= 1);

  // The last index has no freedom just set it and return
  if (dims_remaining == 1) {
    grid.back() = nprocs;
    return;
  }

  // Current index tells us which position we are in
  const ttb_indx current_index = grid.size() - dims_remaining;

  // Make copy for testing on so that we only ever write to grid when we've
  // found a better option
  auto test = grid;
  ttb_indx min_storage = std::numeric_limits<ttb_indx>::max();

  for (auto d : divisors(nprocs)) {
    test[current_index] = d;
    const ttb_indx remaining_procs = nprocs / d;
    recurseMinSpaceGrid(remaining_procs, test, tensor_dims, dims_remaining - 1);

    ttb_indx test_storage = nelementsForRank1Factors(test, tensor_dims);
    if (test_storage < min_storage) {
      min_storage = test_storage;
      grid = test;
    }
  }
}

small_vector<ttb_indx>
minFactorSpaceGrid(ttb_indx nprocs, std::vector<ttb_indx> const &tensor_dims) {
  const ttb_indx ndims = tensor_dims.size();
  small_vector<ttb_indx> grid(ndims);
  if (DistContext::rank() == 0) {
    recurseMinSpaceGrid(nprocs, grid, tensor_dims, ndims);
  }
  DistContext::Bcast(grid, 0);
  return grid;
}

small_vector<ttb_indx>
CartGrid(ttb_indx nprocs, std::vector<ttb_indx> const &tensor_dims) {
  return minFactorSpaceGrid(nprocs, tensor_dims);
}
} // namespace

ProcessorMap::ProcessorMap(std::vector<ttb_indx> const &tensor_dims,
                           small_vector<ttb_indx> const &predetermined_grid,
                           const Dist_Update_Method::type dist_method)
    : dimension_sizes_(predetermined_grid) {
  const ttb_indx ndims = dimension_sizes_.size();

  // I don't think we need to be periodic
  small_vector<int> periodic(ndims, 0);
  bool reorder = true; // Let MPI be smart I guess
  small_vector<int> dimension_sizes_int = convert<int>(dimension_sizes_);
  MPI_Cart_create(DistContext::commWorld(), ndims, dimension_sizes_int.data(),
                  periodic.data(), reorder, &cart_comm_);

  int tmp;
  MPI_Comm_size(cart_comm_, &tmp); grid_nprocs_ = tmp;
  MPI_Comm_rank(cart_comm_, &tmp); grid_rank_ = tmp;
  small_vector<int> coord_int(ndims);
  MPI_Cart_coords(cart_comm_, grid_rank_, ndims, coord_int.data());
  coord_ = convert<ttb_indx>(coord_int);

  small_vector<int> dim_filter(ndims, 1);
  sub_maps_.resize(ndims);
  sub_grid_rank_.resize(ndims);
  sub_comm_sizes_.resize(ndims);

  // Get information for the MPI Subgrid for each Dimension
  for (ttb_indx i = 0; i < ndims; ++i) {
    dim_filter[i] = 0; // Get all dims except this one
    MPI_Cart_sub(cart_comm_, dim_filter.data(), &sub_maps_[i]);
    dim_filter[i] = 1; // Reset the dim_filter

    MPI_Comm_rank(sub_maps_[i], &tmp); sub_grid_rank_[i] = tmp;
    MPI_Comm_size(sub_maps_[i], &tmp); sub_comm_sizes_[i] = tmp;
  }

  small_vector<int> dim_filter2(ndims, 0);
  fac_maps_.resize(ndims);

  // For Tpetra and AllGatherReduce approaches, factor matrices are distributed
  // across all procs
  if (dist_method == Dist_Update_Method::Tpetra ||
      dist_method == Dist_Update_Method::AllGatherReduce ||
      dist_method == Dist_Update_Method::OneSided ||
      dist_method == Dist_Update_Method::TwoSided) {
    for (ttb_indx i = 0; i < ndims; ++i)
      fac_maps_[i] = FacMap(cart_comm_);
  }
  else if (dist_method == Dist_Update_Method::AllReduce){
    // Get information for the MPI Subgrid for each Dimension
    for (ttb_indx i = 0; i < ndims; ++i) {
      dim_filter2[i] = 1; // Get only this dim
      MPI_Comm lcl_comm;
      MPI_Cart_sub(cart_comm_, dim_filter2.data(), &lcl_comm);
      fac_maps_[i] = FacMap(lcl_comm);
      dim_filter2[i] = 0; // Reset the dim_filter
    }
  }
}

ProcessorMap::ProcessorMap(std::vector<ttb_indx> const &tensor_dims,
                           const Dist_Update_Method::type dist_method)
    : ProcessorMap(tensor_dims,
                   CartGrid(DistContext::nranks(), tensor_dims),
                   dist_method) {}

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

#endif
