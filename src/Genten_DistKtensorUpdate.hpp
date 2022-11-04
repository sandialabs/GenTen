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
#include "Genten_Ktensor.hpp"
#include "Genten_IndxArray.hpp"
#include "Genten_Pmap.hpp"
#include "Genten_DistFacMatrix.hpp"
#include "Genten_DistTensorContext.hpp"
#include "Kokkos_UnorderedMap.hpp"

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

  virtual void updateTensor(SptensorT<ExecSpace>& X) {}

  virtual KtensorT<ExecSpace>
  createOverlapKtensor(const KtensorT<ExecSpace>& u) const
  {
    KtensorT<ExecSpace> v = u;
    v.setProcessorMap(nullptr);
    return v;
  };

  virtual void doImport(const KtensorT<ExecSpace>& u_overlapped,
                      const KtensorT<ExecSpace>& u,
                      SystemTimer& timer,
                      const int timer_comm) const
  {
    deep_copy(u_overlapped, u); // no-op if u and u_overlapped are the same
  }

  virtual void doExport(const KtensorT<ExecSpace>& u,
                      const KtensorT<ExecSpace>& u_overlapped,
                      SystemTimer& timer,
                      const int timer_comm,
                      const int timer_update) const = 0;

  virtual void doExport(const KtensorT<ExecSpace>& u,
                      const KtensorT<ExecSpace>& u_overlapped,
                      const ttb_indx n,
                      SystemTimer& timer,
                      const int timer_comm,
                      const int timer_update) const = 0;
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

  virtual void doExport(const KtensorT<ExecSpace>& u,
                      const KtensorT<ExecSpace>& u_overlapped,
                      SystemTimer& timer,
                      const int timer_comm,
                      const int timer_update) const override
  {
    deep_copy(u, u_overlapped); // no-op if u and u_overlapped are the same

    if (pmap != nullptr)
      Kokkos::fence();

    timer.start(timer_comm);
    if (pmap != nullptr) {
      const unsigned nd = u.ndims();
      for (unsigned n=0; n<nd; ++n) {
        auto uv = u[n].view();
        pmap->subGridAllReduce(n, uv.data(), uv.span());
      }
    }
    timer.stop(timer_comm);
  }

  virtual void doExport(const KtensorT<ExecSpace>& u,
                      const KtensorT<ExecSpace>& u_overlapped,
                      const ttb_indx n,
                      SystemTimer& timer,
                      const int timer_comm,
                      const int timer_update) const override
  {
    deep_copy(u[n], u_overlapped[n]); // no-op if u and u_overlapped are the same

    if (pmap != nullptr)
      Kokkos::fence();

    timer.start(timer_comm);
    if (pmap != nullptr) {
      auto uv = u[n].view();
      pmap->subGridAllReduce(n, uv.data(), uv.span());
    }
    timer.stop(timer_comm);
  }

};

template <typename ViewType>
class ViewContainer
{
public:

  typedef typename ViewType::execution_space exec_space;
  typedef Kokkos::View<ViewType*,Kokkos::LayoutRight,exec_space> container_type;
  typedef Kokkos::View<ViewType*,typename ViewType::array_layout,DefaultHostExecutionSpace> host_container_type;
  typedef typename container_type::host_mirror_space::execution_space host_mirror_space;

  // ----- CREATER & DESTROY -----

  // Empty constructor
  KOKKOS_DEFAULTED_FUNCTION
  ViewContainer() = default;

  // Construct an array to hold n factor matrices.
  ViewContainer(ttb_indx n) : data("Genten::ViewContainer::data",n)
  {
    if (Kokkos::Impl::MemorySpaceAccess< typename DefaultHostExecutionSpace::memory_space, typename exec_space::memory_space >::accessible)
      host_data = host_container_type(data.data(), n);
    else
      host_data = host_container_type("Genten::ViewContainer::host_data",n);
  }

  // Copy constructor
  KOKKOS_DEFAULTED_FUNCTION
  ViewContainer(const ViewContainer & src) = default;

  // Destructor.
  KOKKOS_DEFAULTED_FUNCTION
  ~ViewContainer() = default;

  // Make a copy of an existing array.
  KOKKOS_DEFAULTED_FUNCTION
  ViewContainer & operator=(const ViewContainer & src) = default;

  // Set a view
  void set_view(const ttb_indx i, const ViewType& src) const
  {
    assert(i < size());
    host_data[i] = src;
    auto v = data;
    if (!Kokkos::Impl::MemorySpaceAccess< typename DefaultHostExecutionSpace::memory_space, typename exec_space::memory_space >::accessible) {
      Kokkos::RangePolicy<exec_space> policy(0,1);
      Kokkos::parallel_for( policy, KOKKOS_LAMBDA(const ttb_indx j)
      {
        v[i] = src;
      }, "Genten::ViewContainer::set_view");
    }
  }

  // Return the number of factor matrices.
  KOKKOS_INLINE_FUNCTION
  ttb_indx size() const { return data.extent(0); }

  // Return n-th view
  KOKKOS_INLINE_FUNCTION
  const ViewType& operator[](ttb_indx n) const
  {
    KOKKOS_IF_ON_DEVICE(return data[n];)
    KOKKOS_IF_ON_HOST(return host_data[n];)
  }

  // Return n-th view
  KOKKOS_INLINE_FUNCTION
  ViewType& operator[](ttb_indx n)
  {
    KOKKOS_IF_ON_DEVICE(return data[n];)
    KOKKOS_IF_ON_HOST(return host_data[n];)
  }

private:

  // Array of views, each view on device
  container_type data;

  // Host view of array of views
  host_container_type host_data;
};

template <typename ExecSpace>
class KtensorAllGatherUpdate : public DistKtensorUpdate<ExecSpace> {
public:
  typedef Kokkos::View<ttb_indx*, ExecSpace> row_type;
  typedef Kokkos::View<ttb_real**, Kokkos::LayoutRight, ExecSpace> val_type;

private:
  //typedef Kokkos::View<int*, DefaultHostExecutionSpace> count_offset_type;
  typedef std::vector<int> count_offset_type;

  const ProcessorMap *pmap;

  unsigned nd, nc;

  std::vector<int> my_rank;

  ViewContainer<row_type> rows, my_rows;
  std::vector<count_offset_type> row_counts;
  std::vector<count_offset_type> row_offsets;

  ViewContainer<val_type> vals, my_vals;
  std::vector<count_offset_type> val_counts;
  std::vector<count_offset_type> val_offsets;

public:
  KtensorAllGatherUpdate(const KtensorT<ExecSpace>& u,
                         const IndxArray& my_num_updates) :
    pmap(u.getProcessorMap()),
    nd(u.ndims()), nc(u.ncomponents()), my_rank(nd),
    rows(nd), my_rows(nd), row_counts(nd), row_offsets(nd),
    vals(nd), my_vals(nd), val_counts(nd), val_offsets(nd)
  {
    for (unsigned n=0; n<nd; ++n) {

      // Gather row counts from all processors
      if (pmap != nullptr) {
        const int size = pmap->subCommSize(n);
        const int rank = pmap->subCommRank(n);
        my_rank[n] = rank;
        row_counts[n] = count_offset_type(size);
        row_counts[n][rank] = my_num_updates[n];
        pmap->subGridAllGather(n, row_counts[n].data(), 1);
      }
      else {
        my_rank[n] = 0;
        row_counts[n] = count_offset_type(1);
        row_counts[n][0] = my_num_updates[n];
      }
      const int num_proc = row_counts[n].size();

      // Total number of rows
      int num_row = 0;
      for (int p=0; p<num_proc; ++p)
        num_row += row_counts[n][p];

      // Allocate rows and vals
      rows.set_view(n, row_type("row_type_n", num_row));
      vals.set_view(n, val_type("val_type_n", num_row, nc));

      // Compute counts and offsets
      const ttb_indx s = vals[n].stride(0);  // May not equal nc due to padding
      row_offsets[n] = count_offset_type(num_proc);
      val_counts[n] = count_offset_type(num_proc);
      val_offsets[n] = count_offset_type(num_proc);
      val_counts[n][0] = row_counts[n][0]*s;
      row_offsets[n][0] = 0;
      val_offsets[n][0] = 0;
      for (int p=1; p<num_proc; ++p) {
        val_counts[n][p] = row_counts[n][p]*s;
        row_offsets[n][p] = row_offsets[n][p-1] + row_counts[n][p-1];
        val_offsets[n][p] = val_offsets[n][p-1] + val_counts[n][p-1];
      }

      // Get my portion of rows and vals
      const int rank = my_rank[n];
      const ttb_indx my_num_rows = row_counts[n][rank];
      const ttb_indx row_beg = row_offsets[n][rank];
      const ttb_indx row_end = row_beg + my_num_rows;
      const auto row_range = std::make_pair(row_beg, row_end);
      my_rows.set_view(n, Kokkos::subview(rows[n], row_range));
      my_vals.set_view(n, Kokkos::subview(vals[n], row_range, Kokkos::ALL));
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

  virtual void doExport(const KtensorT<ExecSpace>& u,
                      const KtensorT<ExecSpace>& u_overlapped,
                      SystemTimer& timer,
                      const int timer_comm,
                      const int timer_update) const override
  {
    deep_copy(u, u_overlapped); // no-op if u and u_overlapped are the same

    if (pmap != nullptr)
      Kokkos::fence();

    timer.start(timer_comm);
    for (unsigned n=0; n<nd; ++n) {
      auto row_n = rows[n];
      auto val_n = vals[n];

      // Gather contributions from each processor
      if (pmap != nullptr) {
        pmap->subGridAllGather(n, row_n.data(), row_counts[n].data(),
                               row_offsets[n].data());
        pmap->subGridAllGather(n, val_n.data(), val_counts[n].data(),
                               val_offsets[n].data());
      }
    }
    timer.stop(timer_comm);

    // Apply contributions to u
    // In the future, sort vals based on increasing indices in rows and do
    // a thread-local accumulation, which will drastically reduce atomic
    // throughput requirements.  Also use TinyVec.
    timer.start(timer_update);
    u.setMatrices(0.0);
    typedef SpaceProperties<ExecSpace> space_prop;
    for (unsigned n=0; n<nd; ++n) {
      auto row_n = rows[n];
      auto val_n = vals[n];
      auto& u_n = u[n];
      const ttb_indx nrow = row_n.extent(0);
      if (space_prop::concurrency() == 1) {
        Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,nrow),
                             KOKKOS_LAMBDA(const ttb_indx i)
        {
          auto row = row_n[i];
          for (int j=0; j<nc; ++j) {
            u_n.entry(row,j) += val_n(i,j);
          }
        }, "KtensorAllGatherUpdate");
      }
      else {
        Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,nrow),
                             KOKKOS_LAMBDA(const ttb_indx i)
        {
          auto row = row_n[i];
          for (int j=0; j<nc; ++j) {
            Kokkos::atomic_add(&(u_n.entry(row,j)), val_n(i,j));
          }
        }, "KtensorAllGatherUpdate");
      }
    }
    timer.stop(timer_update);
  }

  virtual void doExport(const KtensorT<ExecSpace>& u,
                      const KtensorT<ExecSpace>& u_overlapped,
                      const ttb_indx n,
                      SystemTimer& timer,
                      const int timer_comm,
                      const int timer_update) const override
  {
    assert(u[n].nCols() == nc);

    deep_copy(u[n], u_overlapped[n]); // no-op if u and u_overlapped are the same

    if (pmap != nullptr)
      Kokkos::fence();

    auto row_n = rows[n];
    auto val_n = vals[n];

    // Gather contributions from each processor
    timer.start(timer_comm);
    if (pmap != nullptr) {
      pmap->subGridAllGather(n, row_n.data(), row_counts[n].data(),
                             row_offsets[n].data());
      pmap->subGridAllGather(n, val_n.data(), val_counts[n].data(),
                             val_offsets[n].data());
    }
    timer.stop(timer_comm);

    // Apply contributions to u
    // In the future, sort vals based on increasing indices in rows and do
    // a thread-local accumulation, which will drastically reduce atomic
    // throughput requirements.  Also use TinyVec.
    timer.start(timer_update);
    u[n] = ttb_real(0.0);
    const ttb_indx nrow = row_n.extent(0);
    typedef SpaceProperties<ExecSpace> space_prop;
    if (space_prop::concurrency() == 1) {
      Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,nrow),
                           KOKKOS_LAMBDA(const ttb_indx i)
      {
        auto row = row_n[i];
        for (int j=0; j<nc; ++j) {
          u[n].entry(row,j) += val_n(i,j);
        }
      }, "KtensorAllGatherUpdate");
    }
    else {
      Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,nrow),
                           KOKKOS_LAMBDA(const ttb_indx i)
      {
        auto row = row_n[i];
        for (int j=0; j<nc; ++j) {
          Kokkos::atomic_add(&(u[n].entry(row_n[i],j)), val_n(i,j));
        }
      }, "KtensorAllGatherUpdate");
    }
    timer.stop(timer_update);
  }

  row_type getRowUpdates(const ttb_indx n) const { return my_rows[n]; }
  val_type getValUpdates(const ttb_indx n) const { return my_vals[n]; }

  ViewContainer<row_type> getRowUpdates() const { return my_rows; }
  ViewContainer<val_type> getValUpdates() const { return my_vals; }
};

#ifdef HAVE_TPETRA

template <typename ExecSpace>
class KtensorTpetraUpdate : public DistKtensorUpdate<ExecSpace> {
private:
  DistTensorContext<ExecSpace> dtc;
  std::vector< Teuchos::RCP< tpetra_map_type<ExecSpace> > > overlapMap;
  std::vector< Teuchos::RCP< tpetra_import_type<ExecSpace> > > importer;

public:
  KtensorTpetraUpdate(const DistTensorContext<ExecSpace>& dtc_,
                      const SptensorT<ExecSpace>& X,
                      const KtensorT<ExecSpace>& u) : dtc(dtc_)
  {
    // build importers if needed
    const unsigned nd = u.ndims();
    overlapMap.resize(nd);
    importer.resize(nd);
    if (X.nnz() > 0 && dtc.nprocs() > 1) {
      for (unsigned n=0; n<nd; ++n) {
        overlapMap[n] = dtc.getOverlapFactorMap(n);
        if (!overlapMap[n]->isSameAs(*(dtc.getFactorMap(n))))
          importer[n] = Teuchos::rcp(new tpetra_import_type<ExecSpace>(
            dtc.getFactorMap(n), overlapMap[n]));
      }
    }
  }

  virtual ~KtensorTpetraUpdate() {}

  KtensorTpetraUpdate(KtensorTpetraUpdate&&) = default;
  KtensorTpetraUpdate(const KtensorTpetraUpdate&) = default;
  KtensorTpetraUpdate& operator=(KtensorTpetraUpdate&&) = default;
  KtensorTpetraUpdate& operator=(const KtensorTpetraUpdate&) = default;

  virtual void updateTensor(SptensorT<ExecSpace>& X) override
  {
    using unordered_map_type = Kokkos::UnorderedMap<tpetra_go_type,tpetra_lo_type,DefaultHostExecutionSpace>;

    // Build new overlap maps and importers if needed.
    // For now we are assuming X is subsampled from the original tensor

    if (X.nnz() > 0 && dtc.nprocs() > 1) {
      auto Xh = create_mirror_view(X);
      deep_copy(Xh, X);

      // Build gid -> lid mappings for entries in the new tensor
      // ToDo:  run this on the device
      const unsigned nd = X.ndims();
      const ttb_indx nnz = X.nnz();
      std::vector<unordered_map_type> map(nd);
      std::vector<tpetra_lo_type> cnt(nd, 0);
      for (auto dim=0; dim<nd; ++dim)
        map[dim].rehash(dtc.getOverlapFactorMap(dim)->getLocalNumElements());
      for (ttb_indx i=0; i<nnz; ++i) {
        for (unsigned dim=0; dim<nd; ++dim) {
          const tpetra_lo_type x_lid = Xh.subscript(i,dim);
          const tpetra_go_type gid =
            dtc.getOverlapFactorMap(dim)->getGlobalElement(x_lid);
          auto idx = map[dim].find(gid);
          if (!map[dim].valid_at(idx)) {
            const tpetra_lo_type lid = cnt[dim]++;
            if (map[dim].insert(gid,lid).failed())
              Genten::error("Insertion of GID failed, something is wrong!");
          }
        }
      }
      for (unsigned dim=0; dim<nd; ++dim)
        assert(cnt[dim] == map[dim].size());

      // Construct new overlap maps
      overlapMap.resize(nd);
      const tpetra_go_type indexBase = tpetra_go_type(0);
      const Tpetra::global_size_t invalid =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
      for (auto dim=0; dim<nd; ++dim) {
        Kokkos::View<tpetra_go_type*,DefaultHostExecutionSpace> gids(
          "gids", cnt[dim]);
        const auto sz = map[dim].capacity();
        for (auto idx=0; idx<sz; ++idx) {
          if (map[dim].valid_at(idx)) {
            const auto gid = map[dim].key_at(idx);
            const auto lid = map[dim].value_at(idx);
            gids[lid] = gid;
          }
        }
        overlapMap[dim] = Teuchos::rcp(new tpetra_map_type<ExecSpace>(invalid, gids, indexBase, dtc.getOverlapFactorMap(dim)->getComm()));
      }

      // Build new importers
      for (unsigned n=0; n<nd; ++n)
        if (!overlapMap[n]->isSameAs(*(dtc.getFactorMap(n))))
          importer[n] = Teuchos::rcp(new tpetra_import_type<ExecSpace>(
            dtc.getFactorMap(n), overlapMap[n]));

      // Replace tensor subscripts with new LIDs to match the new LIDs in the
      // overlap maps
      for (ttb_indx i=0; i<nnz; ++i) {
        for (unsigned dim=0; dim<nd; ++dim) {
          const tpetra_lo_type x_lid = Xh.subscript(i,dim);
          const tpetra_go_type gid =
            dtc.getOverlapFactorMap(dim)->getGlobalElement(x_lid);
          const auto idx = map[dim].find(gid);
          const tpetra_lo_type lid = map[dim].value_at(idx);
          Xh.subscript(i,dim) = lid;
        }
      }
      for (unsigned dim=0; dim<nd; ++dim)
        Xh.size()[dim] = cnt[dim];
      deep_copy(X, Xh);
    }
  }

  virtual KtensorT<ExecSpace>
  createOverlapKtensor(const KtensorT<ExecSpace>& u) const override
  {
    KtensorT<ExecSpace> u_overlapped;
    if (dtc.nprocs() == 1) {
      u_overlapped = u;
      u_overlapped.setProcessorMap(nullptr);
    }
    else {
      const unsigned nd = u.ndims();
      const unsigned nc = u.ncomponents();
      u_overlapped = KtensorT<ExecSpace>(nc, nd);
      for (unsigned n=0; n<nd; ++n) {
        FacMatrixT<ExecSpace> mat(overlapMap[n]->getLocalNumElements(), nc);
        u_overlapped.set_factor(n, mat);
      }
    }
    return u_overlapped;
  }

  virtual void doImport(const KtensorT<ExecSpace>& u_overlapped,
                      const KtensorT<ExecSpace>& u,
                      SystemTimer& timer,
                      const int timer_comm) const override
  {
    deep_copy(u_overlapped.weights(), u.weights());
    timer.start(timer_comm);
    const unsigned nd = u.ndims();
    for (unsigned n=0; n<nd; ++n) {
      if (importer[n] != Teuchos::null) {
        DistFacMatrix<ExecSpace> src(u[n], dtc.getFactorMap(n));
        DistFacMatrix<ExecSpace> dst(u_overlapped[n], overlapMap[n]);
        dst.doImport(src, *(importer[n]), Tpetra::INSERT);
      }
      else
        deep_copy(u_overlapped[n], u[n]);
    }
    timer.stop(timer_comm);
  }

  virtual void doExport(const KtensorT<ExecSpace>& u,
                      const KtensorT<ExecSpace>& u_overlapped,
                      SystemTimer& timer,
                      const int timer_comm,
                      const int timer_update) const override
  {
    deep_copy(u.weights(), u_overlapped.weights());

    timer.start(timer_comm);
    const unsigned nd = u.ndims();
    for (unsigned n=0; n<nd; ++n) {
      if (importer[n] != Teuchos::null) {
        DistFacMatrix<ExecSpace> src(u_overlapped[n], overlapMap[n]);
        DistFacMatrix<ExecSpace> dst(u[n], dtc.getFactorMap(n));
        u[n] = ttb_real(0.0);
        dst.doExport(src, *(importer[n]), Tpetra::ADD);
      }
      else
        deep_copy(u[n], u_overlapped[n]);
    }
    timer.stop(timer_comm);
  }

  virtual void doExport(const KtensorT<ExecSpace>& u,
                      const KtensorT<ExecSpace>& u_overlapped,
                      const ttb_indx n,
                      SystemTimer& timer,
                      const int timer_comm,
                      const int timer_update) const override
  {
    timer.start(timer_comm);
    if (importer[n] != Teuchos::null) {
      DistFacMatrix<ExecSpace> src(u_overlapped[n], overlapMap[n]);
      DistFacMatrix<ExecSpace> dst(u[n], dtc.getFactorMap(n));
      u[n] = ttb_real(0.0);
      dst.doExport(src, *(importer[n]), Tpetra::ADD);
    }
    else
      deep_copy(u[n], u_overlapped[n]);
    timer.stop(timer_comm);
  }

};

#endif

}
