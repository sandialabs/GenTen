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

#include "Genten_Util.hpp"

#include "CMakeInclude.h"
#if defined(HAVE_DIST)

#include <vector>

#include "Genten_SmallVector.hpp"
#include "Genten_DistContext.hpp"

namespace Genten {

class ProcessorMap {
public:
  ProcessorMap(std::vector<ttb_indx> const &tensor_dims,
               const Dist_Update_Method::type dist_method);
  ProcessorMap(std::vector<ttb_indx> const &tensor_dims,
               small_vector<ttb_indx> const &predetermined_grid,
               const Dist_Update_Method::type dist_method);
  ~ProcessorMap();

  ProcessorMap() = delete;
  ProcessorMap(ProcessorMap const &) = delete;
  ProcessorMap &operator=(ProcessorMap const &) = delete;

  ProcessorMap(ProcessorMap &&) = delete;
  ProcessorMap &operator=(ProcessorMap &&) = delete;

  // Size of the cartesian grid
  ttb_indx gridSize() const { return grid_nprocs_; }
  small_vector<ttb_indx> const &gridDims() const { return dimension_sizes_; }
  ttb_indx gridDim(ttb_indx dim) const { return dimension_sizes_[dim]; }

  ttb_indx gridRank() const { return grid_rank_; }
  MPI_Comm gridComm() const { return cart_comm_; }

  ttb_indx gridCoord(ttb_indx dim) const { return coord_[dim]; }
  small_vector<ttb_indx> const &gridCoords() const { return coord_; }

  ttb_indx subCommSize(ttb_indx dim) const { return sub_comm_sizes_[dim]; }
  ttb_indx subCommRank(ttb_indx dim) const { return sub_grid_rank_[dim]; }
  MPI_Comm subComm(ttb_indx dim) const { return sub_maps_[dim]; }

  small_vector<ttb_indx> const &subCommSizes() const { return sub_comm_sizes_; }
  small_vector<ttb_indx> const &subCommRanks() const { return sub_grid_rank_; }
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

  template <typename T> void gridAllReduce(T* element, ttb_indx n, MpiOp op = Sum) const {
    static_assert(std::is_arithmetic<T>::value,
                  "gridAllReduce requires something like a double, or int");

    if (grid_nprocs_ > 1) {
      MPI_Allreduce(MPI_IN_PLACE, element, n, DistContext::toMpiType<T>(),
                    convertOp(op), cart_comm_);
    }
  }

  template <typename T> void subGridAllReduce(ttb_indx i, T* element, ttb_indx n, MpiOp op = Sum) const {
    static_assert(std::is_arithmetic<T>::value,
                  "subGridAllReduce requires something like a double, or int");

    if (sub_comm_sizes_[i] > 1) {
      MPI_Allreduce(MPI_IN_PLACE, element, n, DistContext::toMpiType<T>(),
                    convertOp(op), sub_maps_[i]);
    }
  }

  template <typename T> void gridReduce(T* element, ttb_indx n, ttb_indx root, MpiOp op = Sum) const {
    static_assert(std::is_arithmetic<T>::value,
                  "subGridAllReduce requires something like a double, or int");

    if (grid_nprocs_ > 1) {
      if (grid_rank_ == root)
        MPI_Reduce(MPI_IN_PLACE, element, n, DistContext::toMpiType<T>(),
                   convertOp(op), root, cart_comm_);
      else
        MPI_Reduce(element, nullptr, n, DistContext::toMpiType<T>(),
                   convertOp(op), root, cart_comm_);
    }
  }

  template <typename T> void subGridReduce(ttb_indx i, T* element, ttb_indx n, ttb_indx root, MpiOp op = Sum) const {
    static_assert(std::is_arithmetic<T>::value,
                  "subGridAllReduce requires something like a double, or int");

    if (sub_comm_sizes_[i] > 1) {
      if (sub_grid_rank_[i] == root)
        MPI_Reduce(MPI_IN_PLACE, element, n, DistContext::toMpiType<T>(),
                   convertOp(op), root, sub_maps_[i]);
      else
        MPI_Reduce(element, nullptr, n, DistContext::toMpiType<T>(),
                   convertOp(op), root, sub_maps_[i]);
    }
  }

  template <typename ViewT1, typename ViewT2> void subGridReduce(ttb_indx i, const ViewT1& send, const ViewT2& recv, ttb_indx root, MpiOp op = Sum) const {
    using scalar_type1 = typename ViewT1::non_const_value_type;
    using scalar_type2 = typename ViewT2::non_const_value_type;
    static_assert(std::is_same_v<scalar_type1,scalar_type2>,
                  "Views must have the same scalar type.");
    static_assert(std::is_arithmetic<scalar_type1>::value,
                  "subGridAllReduce requires something like a double, or int");
    static_assert(std::is_arithmetic<scalar_type2>::value,
                  "subGridAllReduce requires something like a double, or int");

    scalar_type2* recv_data = nullptr;
    if (sub_grid_rank_[i] == root)
      recv_data = recv.data();
    MPI_Reduce(send.data(), recv_data, send.span(),
               DistContext::toMpiType<scalar_type1>(),
               convertOp(op), root, sub_maps_[i]);
  }

  template <typename T> void gridBcast(T* element, ttb_indx n, ttb_indx root) const {
    static_assert(std::is_arithmetic<T>::value,
                  "gridBcast requires something like a double, or int");

    if (grid_nprocs_ > 1) {
      MPI_Bcast(element, n, DistContext::toMpiType<T>(), root, cart_comm_);
    }
  }

  template <typename T> void subGridBcast(ttb_indx i, T* element, ttb_indx n, ttb_indx root) const {
    static_assert(std::is_arithmetic<T>::value,
                  "subGridBcast requires something like a double, or int");

    if (sub_comm_sizes_[i] > 1) {
      MPI_Bcast(element, n, DistContext::toMpiType<T>(), root, sub_maps_[i]);
    }
  }

  // In-place Allgather
  template <typename T> void gridAllGather(T* element, const ttb_indx count) const {
    static_assert(std::is_arithmetic<T>::value,
                  "gridAllGather requires something like a double, or int");

    if (grid_nprocs_ > 1) {
      MPI_Allgather(MPI_IN_PLACE, count, DistContext::toMpiType<T>(),
                     element, count, DistContext::toMpiType<T>(), cart_comm_);
    }
  }

  // In-place Allgather
  template <typename T> void subGridAllGather(ttb_indx i, T* element, const ttb_indx count) const {
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
  template <typename T> void subGridAllGather(ttb_indx i, T* element, const int counts[], const int offsets[]) const {
    static_assert(std::is_arithmetic<T>::value,
                  "subGridAllGather requires something like a double, or int");

    if (sub_comm_sizes_[i] > 1) {
      // Note, 2nd argument (send_count) is ignored in this version and
      // internally extracted from counts[proc_id]
      MPI_Allgatherv(MPI_IN_PLACE, 0, DistContext::toMpiType<T>(), element,
                     counts, offsets, DistContext::toMpiType<T>(), sub_maps_[i]);
    }
  }

  // Allgatherv
  template <typename ViewT1, typename ViewT2> void subGridAllGather(ttb_indx i, const ViewT1& send, const ViewT2& recv, const int counts[], const int offsets[]) const {
    using scalar_type1 = typename ViewT1::non_const_value_type;
    using scalar_type2 = typename ViewT2::non_const_value_type;
    static_assert(std::is_same_v<scalar_type1,scalar_type2>,
                  "Views must have the same scalar type.");
    static_assert(std::is_arithmetic<scalar_type1>::value,
                  "subGridAllGather requires something like a double, or int");
    static_assert(std::is_arithmetic<scalar_type2>::value,
                  "subGridAllGather requires something like a double, or int");

    MPI_Allgatherv(send.data(), send.span(),
                   DistContext::toMpiType<scalar_type1>(), recv.data(),
                   counts, offsets,
                   DistContext::toMpiType<scalar_type2>(), sub_maps_[i]);
  }

  template <typename T> void gridScan(T* element, ttb_indx n, MpiOp op = Sum) const {
    static_assert(std::is_arithmetic<T>::value,
                  "subGridAllReduce requires something like a double, or int");

    if (grid_nprocs_ > 1) {
      MPI_Scan(MPI_IN_PLACE, element, n, DistContext::toMpiType<T>(),
                 convertOp(op), cart_comm_);
    }
  }

  template <typename T> void subGridScan(ttb_indx i, T* element, ttb_indx n, ttb_indx root, MpiOp op = Sum) const {
    static_assert(std::is_arithmetic<T>::value,
                  "subGridAllReduce requires something like a double, or int");

    if (sub_comm_sizes_[i] > 1) {
      MPI_Scan(MPI_IN_PLACE, element, n, DistContext::toMpiType<T>(),
               convertOp(op), root, sub_maps_[i]);
    }
  }

  // Alltoall
  template <typename T> void subGridAllToAll(
    ttb_indx i,
    const T* send, const ttb_indx count_send,
          T* recv, const ttb_indx count_recv) const {
    static_assert(std::is_arithmetic<T>::value,
                  "subGridAllToAll requires something like a double, or int");

    MPI_Alltoall(
      send, count_send, DistContext::toMpiType<T>(),
      recv, count_recv, DistContext::toMpiType<T>(),
      sub_maps_[i]);
  }

  // Alltoallv
  template <typename T> void subGridAllToAll(
    ttb_indx i,
    const T* send, const int counts_send[], const int offsets_send[],
          T* recv, const int counts_recv[], const int offsets_recv[]) const {
    static_assert(std::is_arithmetic<T>::value,
                  "subGridAllToAll requires something like a double, or int");

    MPI_Alltoallv(
      send, counts_send, offsets_send, DistContext::toMpiType<T>(),
      recv, counts_recv, offsets_recv, DistContext::toMpiType<T>(),
      sub_maps_[i]);
  }

  class FacMap {
  public:
    FacMap() = default;
    FacMap(const FacMap&) = default;
    FacMap(FacMap&&) = default;
    FacMap& operator=(const FacMap&) = default;

    FacMap(MPI_Comm comm) : comm_(comm), size_(0), rank_(0) {
      int tmp;
      MPI_Comm_size(comm_, &tmp); size_ = tmp;
      MPI_Comm_rank(comm_, &tmp); rank_ = tmp;
    }
    ttb_indx size() const { return size_; }
    ttb_indx rank() const { return rank_; }
    template <typename T> T allReduce(T element, MpiOp op = Sum) const {
      static_assert(std::is_arithmetic<T>::value,
                    "allReduce requires something like a double, or int");

      if (size_ > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &element, 1, DistContext::toMpiType<T>(),
                      convertOp(op), comm_);
      }
      return element;
    }

    template <typename T> void allReduce(T* element, ttb_indx n, MpiOp op = Sum) const {
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
    ttb_indx size_;
    ttb_indx rank_;
  };

  const FacMap* facMap(ttb_indx i) const { return &fac_maps_[i]; }

private:
  /*
   * FieldDecls
   */
  MPI_Comm cart_comm_ = MPI_COMM_NULL;
  ttb_indx grid_nprocs_ = 0;
  ttb_indx grid_rank_ = 0;
  small_vector<ttb_indx> coord_;
  small_vector<ttb_indx> dimension_sizes_;

  small_vector<ttb_indx> sub_grid_rank_;
  small_vector<ttb_indx> sub_comm_sizes_;
  small_vector<MPI_Comm> sub_maps_; // N-1 D sub comms

  small_vector<FacMap> fac_maps_; // 1 D sub comms
};

#else

#include "Genten_Kokkos.hpp"

namespace Genten {

class ProcessorMap {
public:
  ProcessorMap() : facMap_() {}
  ~ProcessorMap() = default;

  ProcessorMap(ProcessorMap const &) = delete;
  ProcessorMap &operator=(ProcessorMap const &) = delete;

  ProcessorMap(ProcessorMap &&) = delete;
  ProcessorMap &operator=(ProcessorMap &&) = delete;

  ttb_indx gridSize() const { return 1; }
  ttb_indx gridRank() const { return 0; }
  ttb_indx gridDim(ttb_indx dim) const { return 1; }

  ttb_indx subCommSize(ttb_indx dim) const { return 1; }
  ttb_indx subCommRank(ttb_indx dim) const { return 0; }

  void gridBarrier() const {}

  enum MpiOp {
    Sum,
    Max
  };

  template <typename T> T gridAllReduce(T element, MpiOp op = Sum) const { return element; }
  template <typename T> void gridAllReduce(T* element, ttb_indx n, MpiOp op = Sum) const {}
  template <typename T> void subGridAllReduce(ttb_indx i, T* element, ttb_indx n, MpiOp op = Sum) const {}
  template <typename T> void gridReduce(T* element, ttb_indx n, ttb_indx root, MpiOp op = Sum) const {}
  template <typename T> void subGridReduce(ttb_indx i, T* element, ttb_indx n, ttb_indx root, MpiOp op = Sum) const {}
  template <typename ViewT1, typename ViewT2> void subGridReduce(ttb_indx i, const ViewT1& send, const ViewT2& recv, ttb_indx root, MpiOp op = Sum) const {
    Kokkos::deep_copy(recv, send);
  }
  template <typename T> void gridBcast(T* element, ttb_indx n, ttb_indx root) const {}
  template <typename T> void subGridBcast(ttb_indx i, T* element, ttb_indx n, ttb_indx root) const {}
  template <typename T> void gridAllGather(T* element, const ttb_indx count) const {}
  template <typename T> void subGridAllGather(ttb_indx i, T* element, const ttb_indx count) const {}
  template <typename T> void gridAllGather(T* element, const int counts[], const int offsets[]) const {}
  template <typename T> void subGridAllGather(ttb_indx i, T* element, const int counts[], const int offsets[]) const {}
  template <typename ViewT1, typename ViewT2> void subGridAllGather(ttb_indx i, const ViewT1& send, ViewT2& recv, const int counts[], const int offsets[]) const {
    Kokkos::deep_copy(recv, send);
  }
  template <typename T> void gridScan(T* element, ttb_indx n, MpiOp op = Sum) const {}
  template <typename T> void subGridScan(ttb_indx i, T* element, ttb_indx n, ttb_indx root, MpiOp op = Sum) const {}
  template <typename T> void subGridAllToAll(ttb_indx i,const T* send, const ttb_indx count_send, T* recv, const ttb_indx count_recv) const {
    Genten::error("Not implemented, you shouldn't be calling this!");
  }
  template <typename T> void subGridAllToAll(ttb_indx i, const T* send, const int counts_send[], const int offsets_send[], T* recv, const int counts_recv[], const int offsets_recv[]) const {
    Genten::error("Not implemented, you shouldn't be calling this!");
  }

  class FacMap {
  public:
    FacMap() = default;
    FacMap(const FacMap&) = default;
    FacMap(FacMap&&) = default;
    FacMap& operator=(const FacMap&) = default;

    ttb_indx size() const { return 1; }
    ttb_indx rank() const { return 0; }
    template <typename T> T allReduce(T element, MpiOp op = Sum) const { return element; }
    template <typename T> void allReduce(T* element, ttb_indx n, MpiOp op = Sum) const {}
    template <typename T> void allGather(T* element, const int counts[], const int offsets[]) const {}
  };

  const FacMap* facMap(ttb_indx i) const { return &facMap_; }

private:
  FacMap facMap_;
};

#endif

} // namespace Genten
