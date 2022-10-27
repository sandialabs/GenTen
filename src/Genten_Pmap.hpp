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

#include "CMakeInclude.h"
#if defined(HAVE_DIST)

#include <vector>

#include "Genten_SmallVector.hpp"
#include "Genten_DistContext.hpp"
#include "Genten_Util.hpp"

namespace Genten {

class ProcessorMap {
public:
  ProcessorMap(std::vector<std::uint32_t> const &tensor_dims);
  ProcessorMap(std::vector<std::uint32_t> const &tensor_dims,
               small_vector<int> const &predetermined_grid);
  ~ProcessorMap();

  ProcessorMap() = delete;
  ProcessorMap(ProcessorMap const &) = delete;
  ProcessorMap &operator=(ProcessorMap const &) = delete;

  ProcessorMap(ProcessorMap &&) = delete;
  ProcessorMap &operator=(ProcessorMap &&) = delete;

  // Size of the cartesian grid
  int gridSize() const { return grid_nprocs_; }
  small_vector<int> const &gridDims() const { return dimension_sizes_; }
  int gridDim(int dim) const { return dimension_sizes_[dim]; }

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

  enum MpiOp {
    Sum,
    Max
  };
  static MPI_Op convertOp(MpiOp op) { return op == Sum ? MPI_SUM : MPI_MAX; }

  template <typename T> T gridAllReduce(T element, MpiOp op = Sum) const {
    static_assert(std::is_arithmetic<T>::value,
                  "gridAllReduce requires something like a double, or int");

    if (grid_nprocs_ > 1) {
      MPI_Allreduce(MPI_IN_PLACE, &element, 1, DistContext::toMpiType<T>(),
                    convertOp(op), cart_comm_);
    }
    return element;
  }

  template <typename T> void gridAllReduce(T* element, int n, MpiOp op = Sum) const {
    static_assert(std::is_arithmetic<T>::value,
                  "gridAllReduce requires something like a double, or int");

    if (grid_nprocs_ > 1) {
      MPI_Allreduce(MPI_IN_PLACE, element, n, DistContext::toMpiType<T>(),
                    convertOp(op), cart_comm_);
    }
  }

  template <typename T> void subGridAllReduce(int i, T* element, int n, MpiOp op = Sum) const {
    static_assert(std::is_arithmetic<T>::value,
                  "subGridAllReduce requires something like a double, or int");

    if (sub_comm_sizes_[i] > 1) {
      MPI_Allreduce(MPI_IN_PLACE, element, n, DistContext::toMpiType<T>(),
                    convertOp(op), sub_maps_[i]);
    }
  }

  template <typename T> void gridBcast(T* element, int n, int root) const {
    static_assert(std::is_arithmetic<T>::value,
                  "gridBcast requires something like a double, or int");

    if (grid_nprocs_ > 1) {
      MPI_Bcast(element, n, DistContext::toMpiType<T>(), root, cart_comm_);
    }
  }

  template <typename T> void subGridBcast(int i, T* element, int n, int root) const {
    static_assert(std::is_arithmetic<T>::value,
                  "subGridBcast requires something like a double, or int");

    if (sub_comm_sizes_[i] > 1) {
      MPI_Bcast(element, n, DistContext::toMpiType<T>(), root, sub_maps_[i]);
    }
  }

  // In-place Allgather
  template <typename T> void gridAllGather(T* element, const int count) const {
    static_assert(std::is_arithmetic<T>::value,
                  "gridAllGather requires something like a double, or int");

    if (grid_nprocs_ > 1) {
      MPI_Allgather(MPI_IN_PLACE, count, DistContext::toMpiType<T>(),
                     element, count, DistContext::toMpiType<T>(), cart_comm_);
    }
  }

  // In-place Allgather
  template <typename T> void subGridAllGather(int i, T* element, const int count) const {
    static_assert(std::is_arithmetic<T>::value,
                  "subGridAllGather requires something like a double, or int");

    if (sub_comm_sizes_[i] > 1) {
      MPI_Allgather(MPI_IN_PLACE, count, DistContext::toMpiType<T>(),
                    element, count, DistContext::toMpiType<T>(), sub_maps_[i]);
    }
  }

  // In-place Allgatherv
  template <typename T> void gridAllGather(T* element, const int counts[], const int offsets[]) const {
    static_assert(std::is_arithmetic<T>::value,
                  "gridAllGather requires something like a double, or int");

    if (grid_nprocs_ > 1) {
      // Note, 2nd argument (send_count) is ignored in this version and
      // internally extracted from counts[proc_id]
      MPI_Allgatherv(MPI_IN_PLACE, 0, DistContext::toMpiType<T>(), element,
                     counts, offsets, DistContext::toMpiType<T>(), cart_comm_);
    }
  }

  // In-place Allgatherv
  template <typename T> void subGridAllGather(int i, T* element, const int counts[], const int offsets[]) const {
    static_assert(std::is_arithmetic<T>::value,
                  "subGridAllGather requires something like a double, or int");

    if (sub_comm_sizes_[i] > 1) {
      // Note, 2nd argument (send_count) is ignored in this version and
      // internally extracted from counts[proc_id]
      MPI_Allgatherv(MPI_IN_PLACE, 0, DistContext::toMpiType<T>(), element,
                     counts, offsets, DistContext::toMpiType<T>(), sub_maps_[i]);
    }
  }

  class FacMap {
  public:
    FacMap() = default;
    FacMap(const FacMap&) = default;
    FacMap(FacMap&&) = default;
    FacMap& operator=(const FacMap&) = default;

    FacMap(MPI_Comm comm) : comm_(comm), size_(0), rank_(0) {
      MPI_Comm_size(comm_, &size_);
      MPI_Comm_rank(comm_, &rank_);
    }
    int size() const { return size_; }
    int rank() const { return rank_; }
    template <typename T> T allReduce(T element, MpiOp op = Sum) const {
      static_assert(std::is_arithmetic<T>::value,
                    "allReduce requires something like a double, or int");

      if (size_ > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &element, 1, DistContext::toMpiType<T>(),
                      convertOp(op), comm_);
      }
      return element;
    }

    template <typename T> void allReduce(T* element, int n, MpiOp op = Sum) const {
      static_assert(std::is_arithmetic<T>::value,
                    "allReduce requires something like a double, or int");

      if (size_ > 1) {
        MPI_Allreduce(MPI_IN_PLACE, element, n, DistContext::toMpiType<T>(),
                      convertOp(op), comm_);
      }
    }

    // In-place Allgatherv
      template <typename T> void allGather(T* element, const int counts[], const int offsets[]) const {
        static_assert(std::is_arithmetic<T>::value,
                    "allGather requires something like a double, or int");

        // Note, 2nd argument (send_count) is ignored in this version and
        // internally extracted from counts[proc_id]
        MPI_Allgatherv(MPI_IN_PLACE, 0, DistContext::toMpiType<T>(), element,
                       counts, offsets, DistContext::toMpiType<T>(), comm_);
      }
  private:
    MPI_Comm comm_;
    int size_;
    int rank_;
  };

  const FacMap* facMap(int i) const { return &fac_maps_[i]; }

private:
  /*
   * FieldDecls
   */
  MPI_Comm cart_comm_ = MPI_COMM_NULL;
  int grid_nprocs_ = 0;
  int grid_rank_ = -1;
  small_vector<int> coord_;
  small_vector<int> dimension_sizes_;

  small_vector<int> sub_grid_rank_;
  small_vector<int> sub_comm_sizes_;
  small_vector<MPI_Comm> sub_maps_; // N-1 D sub comms

  small_vector<FacMap> fac_maps_; // 1 D sub comms
};

#else

namespace Genten {

class ProcessorMap {
public:
  ProcessorMap() : facMap_() {}
  ~ProcessorMap() = default;

  ProcessorMap(ProcessorMap const &) = delete;
  ProcessorMap &operator=(ProcessorMap const &) = delete;

  ProcessorMap(ProcessorMap &&) = delete;
  ProcessorMap &operator=(ProcessorMap &&) = delete;

  int gridSize() const { return 1; }
  int gridRank() const { return 0; }
  int gridDim(int dim) const { return 1; }

  int subCommSize(int dim) const { return 1; }
  int subCommRank(int dim) const { return 0; }

  void gridBarrier() const {}

  enum MpiOp {
    Sum,
    Max
  };

  template <typename T> T gridAllReduce(T element, MpiOp op = Sum) const { return element; }
  template <typename T> void gridAllReduce(T* element, int n, MpiOp op = Sum) const {}
  template <typename T> void subGridAllReduce(int i, T* element, int n, MpiOp op = Sum) const {}
  template <typename T> void gridBcast(T* element, int n, int root) const {}
  template <typename T> void subGridBcast(int i, T* element, int n, int root) const {}

  class FacMap {
  public:
    FacMap() = default;
    FacMap(const FacMap&) = default;
    FacMap(FacMap&&) = default;
    FacMap& operator=(const FacMap&) = default;

    int size() const { return 1; }
    int rank() const { return 0; }
    template <typename T> T allReduce(T element, MpiOp op = Sum) const { return element; }
    template <typename T> void allReduce(T* element, int n, MpiOp op = Sum) const {}
  };

  const FacMap* facMap(int i) const { return &facMap_; }

private:
  FacMap facMap_;
};

#endif

} // namespace Genten
