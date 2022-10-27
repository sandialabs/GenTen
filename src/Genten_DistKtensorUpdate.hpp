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

#include "Genten_Ktensor.hpp"
#include "Genten_IndxArray.hpp"
#include "Genten_Pmap.hpp"

namespace Genten {

template <typename ExecSpace>
class DistKtensorUpdate {
public:
  DistKtensorUpdate() = default;
  virtual ~DistKtensorUpdate() {}

  DistKtensorUpdate(DistKtensorUpdate&&) = default;
  DistKtensorUpdate(const DistKtensorUpdate&) = default;
  DistKtensorUpdate& operator=(DistKtensorUpdate&&) = default;
  DistKtensorUpdate& operator=(const DistKtensorUpdate&) = default;

  virtual void update(const KtensorT<ExecSpace>& u) const
  {
    const ttb_indx nd = u.ndims();
    for (ttb_indx n=0; n<nd; ++n)
      update(u[n], n);
  }
  virtual void update(const FacMatrixT<ExecSpace>& u, const ttb_indx n) const = 0;
};

template <typename ExecSpace>
class KtensorAllReduceUpdate : public DistKtensorUpdate<ExecSpace> {
private:
  const ProcessorMap *pmap;

public:
  KtensorAllReduceUpdate(const KtensorT<ExecSpace>& u) :
    pmap(u.getProcessorMap()) {}
  virtual ~KtensorAllReduceUpdate() {}

  KtensorAllReduceUpdate(KtensorAllReduceUpdate&&) = default;
  KtensorAllReduceUpdate(const KtensorAllReduceUpdate&) = default;
  KtensorAllReduceUpdate& operator=(KtensorAllReduceUpdate&&) = default;
  KtensorAllReduceUpdate& operator=(const KtensorAllReduceUpdate&) = default;

  virtual void update(const FacMatrixT<ExecSpace>& u, const ttb_indx n) const
  {
    if (pmap != nullptr) {
      Kokkos::fence();
      auto uv = u.view();
      pmap->subGridAllReduce(n, uv.data(), uv.span());
    }
  }

};

template <typename ExecSpace>
class KtensorAllGatherUpdate : public DistKtensorUpdate<ExecSpace> {
public:
  typedef Kokkos::View<ttb_indx*, ExecSpace> row_type;
  typedef Kokkos::View<ttb_real**, Kokkos::LayoutRight, ExecSpace> val_type;

private:
  typedef Kokkos::View<int*, DefaultHostExecutionSpace> count_offset_type;

  const ProcessorMap *pmap;

  int nd, nc;

  std::vector<int> my_rank;

  std::vector<row_type> rows;
  std::vector<count_offset_type> row_counts;
  std::vector<count_offset_type> row_offsets;

  std::vector<val_type> vals;
  std::vector<count_offset_type> val_counts;
  std::vector<count_offset_type> val_offsets;

public:
  KtensorAllGatherUpdate(const KtensorT<ExecSpace>& u,
                         const IndxArray& my_num_updates) :
    pmap(u.getProcessorMap()),
    nd(u.ndims()), nc(u.ncomponents()), my_rank(nd),
    rows(nd), row_counts(nd), row_offsets(nd),
    vals(nd), val_counts(nd), val_offsets(nd)
  {
    for (int n=0; n<nd; ++n) {

      // Gather row counts from all processors
      if (pmap != nullptr) {
        const int size = pmap->subCommSize(n);
        const int rank = pmap->subCommRank(n);
        my_rank[n] = rank;
        row_counts[n] = count_offset_type("row_counts_n", size);
        row_counts[n](rank) = my_num_updates[n];
        pmap->subGridAllGather(n, row_counts[n].data(), 1);
      }
      else {
        my_rank[n] = 0;
        row_counts[n] = count_offset_type("row_counts_n", 1);
        row_counts[n](0) = my_num_updates[n];
      }
      const int num_proc = row_counts[n].size();

      // Total number of rows
      int num_row = 0;
      for (int p=0; p<num_proc; ++p)
        num_row += row_counts[n](p);

      // Allocate row and val
      rows[n] = row_type("row_type_n", num_row);
      vals[n] = val_type("val_type_n", num_row, nc);

      // Compute counts and offsets
      const ttb_indx s = vals[n].stride(0);  // May not equal nc due to padding
      row_offsets[n] = count_offset_type("row_offsets_n", num_proc);
      val_counts[n] = count_offset_type("val_counts_n", num_proc);
      val_offsets[n] = count_offset_type("val_offsets_n", num_proc);
      val_counts[n](0) = row_counts[n](0)*s;
      row_offsets[n](0) = 0;
      val_offsets[n](0) = 0;
      for (int p=1; p<num_proc; ++p) {
        val_counts[n](p) = row_counts[n](p)*s;
        row_offsets[n](p) = row_offsets[n](p-1) + row_counts[n](p-1);
        val_offsets[n](p) = val_offsets[n](p-1) + val_counts[n](p-1);
      }
    }
  }
  KtensorAllGatherUpdate(const KtensorT<ExecSpace>& u,
                         const ttb_indx my_num_updates) :
    KtensorAllGatherUpdate(u, IndxArray(u.ndims(), my_num_updates)) {}
  virtual ~KtensorAllGatherUpdate() {}

  KtensorAllGatherUpdate(KtensorAllGatherUpdate&&) = default;
  KtensorAllGatherUpdate(const KtensorAllGatherUpdate&) = default;
  KtensorAllGatherUpdate& operator=(KtensorAllGatherUpdate&&) = default;
  KtensorAllGatherUpdate& operator=(const KtensorAllGatherUpdate&) = default;

  virtual void update(const FacMatrixT<ExecSpace>& u, const ttb_indx n) const
  {
    assert(u.nCols() == nc);

    if (pmap != nullptr)
      Kokkos::fence();

    auto row_n = rows[n];
    auto val_n = vals[n];

    // Gather contributions from each processor
    if (pmap != nullptr) {
      pmap->subGridAllGather(n, row_n.data(), row_counts[n].data(),
                             row_offsets[n].data());
      pmap->subGridAllGather(n, val_n.data(), val_counts[n].data(),
                             val_offsets[n].data());
    }

    // Apply contributions to u
    // In the future, sort vals based on increasing indices in rows and do
    // a thread-local accumulation, which will drastically reduce atomic
    // throughput requirements.  Also use TinyVec.
    u = ttb_real(0.0);
    const ttb_indx nrow = row_n.extent(0);
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,nrow),
                         KOKKOS_LAMBDA(const ttb_indx i)
    {
      //typedef SpaceProperties<ExecSpace> space_prop;
      auto row = row_n[i];
      for (int j=0; j<nc; ++j) {
        // if (space_prop::concurrency() > 1)
        //   Kokkos::atomic_add(&(u.entry(row_n[i],j)), val_n(i,j));
        // else
          u.entry(row,j) += val_n(i,j);
      }
    }, "KtensorAllGatherUpdate");
  }

  row_type getRowUpdates(const ttb_indx n) const
  {
    const int rank = my_rank[n];
    const ttb_indx num_rows = row_counts[n][rank];
    const ttb_indx row_beg = row_offsets[n][rank];
    const ttb_indx row_end = row_beg + num_rows;
    const auto row_range = std::make_pair(row_beg, row_end);
    return Kokkos::subview(rows[n], row_range);
  }
  val_type getValUpdates(const ttb_indx n) const
  {
    const int rank = my_rank[n];
    const ttb_indx num_rows = row_counts[n][rank];
    const ttb_indx row_beg = row_offsets[n][rank];
    const ttb_indx row_end = row_beg + num_rows;
    const auto row_range = std::make_pair(row_beg, row_end);
    return Kokkos::subview(vals[n], row_range, Kokkos::ALL);
  }

  void setMyUpdates(const std::vector<row_type>& my_rows,
                    const std::vector<val_type>& my_vals)
  {
    assert(my_rows.size() == nd);
    assert(my_vals.size() == nd);

    for (int n=0; n<nd; ++n)
      setMyUpdates(my_rows[n], my_vals[n], n);
  }

  void setMyUpdates(const row_type& my_rows, const val_type& my_vals,
                    const ttb_indx n)
  {
    const int rank = my_rank[n];
    const ttb_indx num_rows = row_counts[n][rank];
    assert(my_rows[n].extent(0) == num_rows);
    assert(my_vals[n].extent(0) == num_rows);
    assert(my_vals[n].extent(1) == nc);

    const ttb_indx row_beg = row_offsets[n][rank];
    const ttb_indx row_end = row_beg + num_rows;
    const auto row_range = std::make_pair(row_beg, row_end);
    auto row = Kokkos::subview(rows[n], row_range);
    auto val = Kokkos::subview(vals[n], row_range, Kokkos::ALL);
    Kokkos::deep_copy(row, my_rows);
    Kokkos::deep_copy(val, my_vals);
  }
};

}
