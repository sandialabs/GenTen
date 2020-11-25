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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

namespace Genten {

namespace {
// Silly function to compute divisors
auto divisors(int input) {
  std::vector<int> divisors;
  int sroot = std::sqrt(input);
  for (auto i = 1; i <= sroot; ++i) {
    if (input % i == 0) {
      divisors.push_back(i);
      if (i != sroot) {
        divisors.push_back(input / i);
      }
    }
  }

  std::sort(divisors.begin(), divisors.end());
  return divisors;
}

auto spaceRankMultiplierForFactors(std::vector<int> const &grid,
                                   std::vector<int> const &tensor_dims) {
  const auto ndims = grid.size();
  std::vector<int64_t> SizeNotMe(ndims, 1);
  // Just do this the slow way for now, it can be faster if really needed
  for (auto i = 0; i < ndims; ++i) {
    for (auto j = 0; j < ndims; ++j) {
      if (j != i) {
        SizeNotMe[i] *= grid[j];
      }
    }
  }

  int64_t total_element_multiplier = 0;
  for (auto i = 0; i < ndims; ++i) {
    total_element_multiplier += SizeNotMe[i] * int64_t(tensor_dims[i]);
  }

  return total_element_multiplier;
}

auto minFactorSpaceGrid(int nprocs, std::vector<int> const &tensor_dims) {
  std::vector<int> grid_dims(tensor_dims.size());

  // Let's just do a brute force thing and see how it goes
  int64_t smallest_nel = std::numeric_limits<int64_t>::max();
  std::vector<int> best_grid;

  for (auto d0 : divisors(nprocs)) {
    grid_dims[0] = d0;

    for (auto d1 : divisors(nprocs / d0)) {
      const auto d01 = d0 * d1;
      grid_dims[1] = d1;
      grid_dims[2] = nprocs/d01;

      // for (auto d2 : divisors(nprocs / d01)) {
      //   const auto d012 = d01 * d2;
      //   grid_dims[2] = d2;

      //   for (auto d3 : divisors(nprocs / d012)) {
      //     const auto d0123 = d012 * d3;
      //     grid_dims[3] = d3;
      //     grid_dims[4] = nprocs / d0123;

          if (std::accumulate(grid_dims.begin(), grid_dims.end(), 1,
                              std::multiplies<int>()) != nprocs) {
            std::cout << "Oops I messed up.\n";
            break;
          }

          auto nelems = spaceRankMultiplierForFactors(grid_dims, tensor_dims);
          if (nelems < smallest_nel) {
            std::cout << "Grid: ";
            for (auto g : grid_dims) {
              std::cout << g << " ";
            }
            std::cout << std::endl;
            std::cout << "\tTotal GB with rank 50: " << nelems * 50 * 8 * 1e-9
                      << "\n";
            smallest_nel = nelems;
            best_grid = grid_dims;
          }
      //   }
     //  }
    }
  }

  return best_grid;
}

auto minAllReduceComm(int nprocs, std::vector<int> const &tensor_dims) {
  std::vector<int> grid_dims(tensor_dims.size());
  return grid_dims;
}

enum class CartGridStratagy { MinAllReduceComm, MinFactorSpace };

auto CartGrid(int nprocs, std::vector<int> const &tensor_dims,
              CartGridStratagy strat = CartGridStratagy::MinAllReduceComm) {
  switch (strat) {
  case CartGridStratagy::MinAllReduceComm:
    return minAllReduceComm(nprocs, tensor_dims);
  case CartGridStratagy::MinFactorSpace:
    return minFactorSpaceGrid(nprocs, tensor_dims);
  }
}
} // namespace

ProcessorMap::ProcessorMap(ptree const &input_tree, TensorInfo const &info)
    : tensor_info_(info), pmap_tree_(input_tree.get_child("pmap", ptree{})) {
  if (DistContext::rank() == 0) {

    if (auto file_name = input_tree.get_optional<std::string>("tensor.file")) {
      std::ifstream tensor_file(file_name.value());
      auto header_info = read_sptensor_header(tensor_file);

      std::cout << "Dimensions of tensor: ";
      for (auto d : header_info.dim_sizes) {
        std::cout << d << " ";
      }
      std::cout << std::endl;

      // Fake with 1000 for now
      auto grid = CartGrid(10000, header_info.dim_sizes,
                           CartGridStratagy::MinFactorSpace);

    } else {
      std::cout << "No tensor file\n";
    }
  }
}
} // namespace Genten
