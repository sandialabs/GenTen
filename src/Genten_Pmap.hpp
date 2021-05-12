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

#pragma once

#include "Genten_Boost.hpp"
#include "Genten_DistContext.hpp"
#include "Genten_TensorInfo.hpp"
#include "Genten_Util.hpp"

namespace Genten {

class ProcessorMap {
public:
  ProcessorMap(ptree const &input_tree, TensorInfo const &Ti);
  ~ProcessorMap();

  ProcessorMap() = delete;
  ProcessorMap(ProcessorMap const &) = delete;
  ProcessorMap &operator=(ProcessorMap const &) = delete;

  ProcessorMap(ProcessorMap &&) = delete;
  ProcessorMap &operator=(ProcessorMap &&) = delete;

  // Size of the cartesian grid
  int gridSize() const { return grid_nprocs_; }
  small_vector<int> const &gridDims() const { return dimension_sizes_; }

  int gridRank() const { return grid_rank_; }
  MPI_Comm gridComm() const { return cart_comm_; }

  int gridCoord(int dim) const { return coord_[dim]; }
  small_vector<int> const &gridCoords() const { return coord_; }

  int subCommSize(int dim) const { return sub_comm_sizes_[dim]; }
  int subCommRank(int dim) const { return sub_grid_rank_[dim]; }
  MPI_Comm subComm(int dim) const { return sub_maps_[dim]; }

  small_vector<int> const &subCommSizes() const { return sub_comm_sizes_; }
  small_vector<int> const &subCommRanks() const { return sub_grid_rank_; }
  small_vector<MPI_Comm> const &subComms() const { return sub_maps_; }

  void gridBarrier() const;

  template <typename T> T gridAllReduce(T element, MPI_Op op = MPI_SUM) const {
    static_assert(std::is_arithmetic<T>::value,
                  "gridAllReduce requires something like a double, or int");

    if(grid_nprocs_ > 1){
      MPI_Allreduce(MPI_IN_PLACE, &element, 1, convertType(element), op,
                    cart_comm_);
    }
    return element;
  }

private:
  MPI_Datatype convertType(double) const { return MPI_DOUBLE; }

  MPI_Datatype convertType(float) const { return MPI_FLOAT; }

  MPI_Datatype convertType(unsigned long) const { return MPI_UNSIGNED_LONG; }

  MPI_Datatype convertType(int) const { return MPI_INT; }

  MPI_Datatype convertType(long) const { return MPI_LONG; }

private:
  /*
   * FieldDecls
   */
  MPI_Comm cart_comm_ = MPI_COMM_NULL;
  int grid_nprocs_;
  int grid_rank_;
  small_vector<int> coord_;
  small_vector<int> dimension_sizes_;

  small_vector<int> sub_grid_rank_;
  small_vector<int> sub_comm_sizes_;
  small_vector<MPI_Comm> sub_maps_; // N-1 D sub comms

  ptree pmap_tree_;
};

} // namespace Genten
