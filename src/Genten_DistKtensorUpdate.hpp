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
#include "Genten_DistTensor.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_IndxArray.hpp"
#include "Genten_Pmap.hpp"
#include "Genten_DistFacMatrix.hpp"
#include "Genten_SystemTimer.hpp"

#include "Kokkos_UnorderedMap.hpp"
#include <unordered_map>

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

  virtual void updateTensor(const DistTensor<ExecSpace>& X) {}

  virtual KtensorT<ExecSpace>
  createOverlapKtensor(const KtensorT<ExecSpace>& u) const
  {
    return u;
  };

  virtual void initOverlapKtensor(KtensorT<ExecSpace>& u) const
  {
    GENTEN_TIME_MONITOR("k-tensor init");
    u.weights() = ttb_real(1.0);
    u.setMatrices(0.0);
  }

  virtual bool overlapAliasesArg() const { return true; }

  virtual bool isReplicated() const { return true; }

  virtual bool overlapDependsOnTensor() const { return false; }

  virtual void doImport(const KtensorT<ExecSpace>& u_overlapped,
                        const KtensorT<ExecSpace>& u) const
  {
    deep_copy(u_overlapped, u); // no-op if u and u_overlapped are the same
  }

  virtual void doImport(const KtensorT<ExecSpace>& u_overlapped,
                        const KtensorT<ExecSpace>& u,
                        const ttb_indx n) const
  {
    deep_copy(u_overlapped[n], u[n]); // no-op if u and u_overlapped are the same
  }

  virtual void doImport(const KtensorT<ExecSpace>& u_overlapped,
                        const KtensorT<ExecSpace>& u,
                        SystemTimer& timer,
                        const int timer_comm) const
  {
    timer.start(timer_comm);
    this->doImport(u_overlapped, u);
    timer.stop(timer_comm);
  }

  virtual void doImport(const KtensorT<ExecSpace>& u_overlapped,
                        const KtensorT<ExecSpace>& u,
                        const ttb_indx n,
                        SystemTimer& timer,
                        const int timer_comm) const
  {
    timer.start(timer_comm);
    this->doImport(u_overlapped, u, n);
    timer.stop(timer_comm);
  }

  virtual void doExport(const KtensorT<ExecSpace>& u,
                        const KtensorT<ExecSpace>& u_overlapped) const = 0;

  virtual void doExport(const KtensorT<ExecSpace>& u,
                        const KtensorT<ExecSpace>& u_overlapped,
                        const ttb_indx n) const = 0;

  virtual void doExport(const KtensorT<ExecSpace>& u,
                        const KtensorT<ExecSpace>& u_overlapped,
                        SystemTimer& timer,
                        const int timer_comm,
                        const int timer_update) const
  {
    timer.start(timer_comm);
    this->doExport(u, u_overlapped);
    timer.stop(timer_comm);
  }

  virtual void doExport(const KtensorT<ExecSpace>& u,
                        const KtensorT<ExecSpace>& u_overlapped,
                        const ttb_indx n,
                        SystemTimer& timer,
                        const int timer_comm,
                        const int timer_update) const
  {
    timer.start(timer_comm);
    this->doExport(u, u_overlapped, n);
    timer.stop(timer_comm);
  }
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

  using DistKtensorUpdate<ExecSpace>::doExport;

  virtual void doExport(const KtensorT<ExecSpace>& u,
                        const KtensorT<ExecSpace>& u_overlapped) const override
  {
    GENTEN_TIME_MONITOR("k-tensor export");

    deep_copy(u, u_overlapped); // no-op if u and u_overlapped are the same

    if (pmap != nullptr)
      Kokkos::fence();

    if (pmap != nullptr) {
      const unsigned nd = u.ndims();
      for (unsigned n=0; n<nd; ++n) {
        auto uv = u[n].view();
        pmap->subGridAllReduce(n, uv.data(), uv.span());
      }
    }
  }

  virtual void doExport(const KtensorT<ExecSpace>& u,
                        const KtensorT<ExecSpace>& u_overlapped,
                        const ttb_indx n) const override
  {
    GENTEN_TIME_MONITOR("k-tensor export");

    deep_copy(u[n], u_overlapped[n]); // no-op if u and u_overlapped are the same

    if (pmap != nullptr)
      Kokkos::fence();

    if (pmap != nullptr) {
      auto uv = u[n].view();
      pmap->subGridAllReduce(n, uv.data(), uv.span());
    }
  }

};

#ifdef HAVE_TPETRA

template <typename ExecSpace>
class KtensorTpetraUpdate : public DistKtensorUpdate<ExecSpace> {
private:
  DistTensor<ExecSpace> X;

public:
  KtensorTpetraUpdate(const DistTensor<ExecSpace>& X_,
                      const KtensorT<ExecSpace>& u) : X(X_) {}

  virtual ~KtensorTpetraUpdate() {}

  KtensorTpetraUpdate(KtensorTpetraUpdate&&) = default;
  KtensorTpetraUpdate(const KtensorTpetraUpdate&) = default;
  KtensorTpetraUpdate& operator=(KtensorTpetraUpdate&&) = default;
  KtensorTpetraUpdate& operator=(const KtensorTpetraUpdate&) = default;

  virtual bool overlapDependsOnTensor() const override { return true; }

  virtual void updateTensor(const DistTensor<ExecSpace>& X_) override
  {
    X = X_;
  }

  virtual KtensorT<ExecSpace>
  createOverlapKtensor(const KtensorT<ExecSpace>& u) const override
  {
    GENTEN_TIME_MONITOR("create overlapped k-tensor");

    const unsigned nd = u.ndims();
    const unsigned nc = u.ncomponents();
    KtensorT<ExecSpace> u_overlapped = KtensorT<ExecSpace>(nc, nd);
    for (unsigned n=0; n<nd; ++n) {
      FacMatrixT<ExecSpace> mat(X.tensorMap(n)->getLocalNumElements(), nc);
      u_overlapped.set_factor(n, mat);
    }
    u_overlapped.setProcessorMap(u.getProcessorMap());
    return u_overlapped;
  }

  virtual bool overlapAliasesArg() const override { return false; }

  virtual bool isReplicated() const override { return false; }

  using DistKtensorUpdate<ExecSpace>::doImport;

  virtual void doImport(const KtensorT<ExecSpace>& u_overlapped,
                        const KtensorT<ExecSpace>& u) const override
  {
    GENTEN_TIME_MONITOR("k-tensor import");

    deep_copy(u_overlapped.weights(), u.weights());
    const unsigned nd = u.ndims();
    for (unsigned n=0; n<nd; ++n) {
      if (X.importer(n) != Teuchos::null) {
        DistFacMatrix<ExecSpace> src(u[n], X.factorMap(n));
        DistFacMatrix<ExecSpace> dst(u_overlapped[n], X.tensorMap(n));
        dst.doImport(src, *(X.importer(n)), Tpetra::INSERT);
      }
      else
        deep_copy(u_overlapped[n], u[n]);
    }
  }

  virtual void doImport(const KtensorT<ExecSpace>& u_overlapped,
                        const KtensorT<ExecSpace>& u,
                        const ttb_indx n) const override
  {
    GENTEN_TIME_MONITOR("k-tensor import");

    if (X.importer(n) != Teuchos::null) {
      DistFacMatrix<ExecSpace> src(u[n], X.factorMap(n));
      DistFacMatrix<ExecSpace> dst(u_overlapped[n], X.tensorMap(n));
      dst.doImport(src, *(X.importer(n)), Tpetra::INSERT);
    }
    else
      deep_copy(u_overlapped[n], u[n]);
  }

  using DistKtensorUpdate<ExecSpace>::doExport;

  virtual void doExport(const KtensorT<ExecSpace>& u,
                        const KtensorT<ExecSpace>& u_overlapped) const override
  {
    GENTEN_TIME_MONITOR("k-tensor export");

    deep_copy(u.weights(), u_overlapped.weights());

    const unsigned nd = u.ndims();
    for (unsigned n=0; n<nd; ++n) {
      if (X.importer(n) != Teuchos::null) {
        DistFacMatrix<ExecSpace> src(u_overlapped[n], X.tensorMap(n));
        DistFacMatrix<ExecSpace> dst(u[n], X.factorMap(n));
        u[n] = ttb_real(0.0);
        dst.doExport(src, *(X.importer(n)), Tpetra::ADD);
      }
      else
        deep_copy(u[n], u_overlapped[n]);
    }
  }

  virtual void doExport(const KtensorT<ExecSpace>& u,
                        const KtensorT<ExecSpace>& u_overlapped,
                        const ttb_indx n) const override
  {
    GENTEN_TIME_MONITOR("k-tensor export");

    if (X.importer(n) != Teuchos::null) {
      DistFacMatrix<ExecSpace> src(u_overlapped[n], X.tensorMap(n));
      DistFacMatrix<ExecSpace> dst(u[n], X.factorMap(n));
      u[n] = ttb_real(0.0);
      dst.doExport(src, *(X.importer(n)), Tpetra::ADD);
    }
    else
      deep_copy(u[n], u_overlapped[n]);
  }

};

#endif

template <typename ExecSpace>
class KtensorAllGatherReduceUpdate : public DistKtensorUpdate<ExecSpace> {
private:
  const ProcessorMap *pmap;
  std::vector< std::vector<int> > offsets;
  std::vector< std::vector<int> > sizes;

  std::vector< std::vector<int> > offsets_r;
  std::vector< std::vector<int> > sizes_r;

public:
  KtensorAllGatherReduceUpdate(const KtensorT<ExecSpace>& u) :
    pmap(u.getProcessorMap())
  {
    const unsigned nd = u.ndims();
    sizes.resize(nd);
    sizes_r.resize(nd);
    offsets.resize(nd);
    offsets_r.resize(nd);
    for (unsigned n=0; n<nd; ++n) {
      if (pmap != nullptr) {
        const unsigned np = pmap->subCommSize(n);

        // Get number of rows on each processor
        sizes[n].resize(np);
        sizes[n][pmap->subCommRank(n)] = u[n].nRows();
        pmap->subGridAllGather(n, sizes[n].data(), 1);

        // Get span on each processor
        sizes_r[n].resize(np);
        sizes_r[n][pmap->subCommRank(n)] = u[n].view().span();
        pmap->subGridAllGather(n, sizes_r[n].data(), 1);

        // Get starting offsets for each processor
        offsets[n].resize(np);
        offsets_r[n].resize(np);
        offsets[n][0] = 0;
        offsets_r[n][0] = 0;
        for (unsigned p=1; p<np; ++p) {
          offsets[n][p] = offsets[n][p-1] + sizes[n][p-1];
          offsets_r[n][p] = offsets_r[n][p-1] + sizes_r[n][p-1];
        }
      }
      else {
        sizes[n].resize(1);
        sizes_r[n].resize(1);
        offsets[n].resize(1);
        offsets_r[n].resize(1);
        sizes[n][0] = u[n].nRows();
        sizes_r[n][0] = u[n].view().span();
        offsets[n][0] = 0;
        offsets_r[n][0] = 0;
      }
    }
  }
  virtual ~KtensorAllGatherReduceUpdate() {}

  KtensorAllGatherReduceUpdate(KtensorAllGatherReduceUpdate&&) = default;
  KtensorAllGatherReduceUpdate(const KtensorAllGatherReduceUpdate&) = default;
  KtensorAllGatherReduceUpdate& operator=(KtensorAllGatherReduceUpdate&&) = default;
  KtensorAllGatherReduceUpdate& operator=(const KtensorAllGatherReduceUpdate&) = default;

  virtual KtensorT<ExecSpace>
  createOverlapKtensor(const KtensorT<ExecSpace>& u) const override
  {
    GENTEN_TIME_MONITOR("create overlapped k-tensor");

    const unsigned nd = u.ndims();
    const unsigned nc = u.ncomponents();
    KtensorT<ExecSpace> u_overlapped = KtensorT<ExecSpace>(nc, nd);
    for (unsigned n=0; n<nd; ++n) {
      const unsigned np = offsets[n].size();
      const ttb_indx nrows = offsets[n][np-1]+sizes[n][np-1];
      FacMatrixT<ExecSpace> mat(nrows, nc);
      u_overlapped.set_factor(n, mat);
    }
    u_overlapped.setProcessorMap(u.getProcessorMap());
    return u_overlapped;
  };

  virtual bool overlapAliasesArg() const override { return false; }

  virtual bool isReplicated() const override { return false; }

  using DistKtensorUpdate<ExecSpace>::doImport;

  virtual void doImport(const KtensorT<ExecSpace>& u_overlapped,
                        const KtensorT<ExecSpace>& u) const override
  {
    GENTEN_TIME_MONITOR("k-tensor import");

    if (pmap != nullptr) {
      const unsigned nd = u.ndims();
      for (unsigned n=0; n<nd; ++n) {
        auto uov = u_overlapped[n].view();
        const unsigned rank = pmap->subCommRank(n);
        const unsigned np = pmap->subCommSize(n);
        gt_assert(u[n].view().span() == size_t(sizes_r[n][rank]));
        gt_assert(uov.span() == size_t(offsets_r[n][np-1]+sizes_r[n][np-1]));
        Kokkos::fence();
        pmap->subGridAllGather(n, u[n].view(), uov,
                               sizes_r[n].data(), offsets_r[n].data());
      }
    }
    else
      deep_copy(u_overlapped, u); // no-op if u and u_overlapped are the same
  }

  virtual void doImport(const KtensorT<ExecSpace>& u_overlapped,
                        const KtensorT<ExecSpace>& u,
                        const ttb_indx n) const override
  {
    GENTEN_TIME_MONITOR("k-tensor import");

    if (pmap != nullptr) {
      auto uov = u_overlapped[n].view();
      const unsigned rank = pmap->subCommRank(n);
      const unsigned np = pmap->subCommSize(n);
      gt_assert(u[n].view().span() == size_t(sizes_r[n][rank]));
      gt_assert(uov.span() == size_t(offsets_r[n][np-1]+sizes_r[n][np-1]));
      Kokkos::fence();
      pmap->subGridAllGather(n, u[n].view(), uov,
                             sizes_r[n].data(), offsets_r[n].data());
    }
    else
      deep_copy(u_overlapped[n], u[n]); // no-op if u and u_overlapped are the same
  }

  using DistKtensorUpdate<ExecSpace>::doExport;

  virtual void doExport(const KtensorT<ExecSpace>& u,
                        const KtensorT<ExecSpace>& u_overlapped) const override
  {
    GENTEN_TIME_MONITOR("k-tensor export");

    if (pmap != nullptr) {
      const unsigned nd = u.ndims();
      for (unsigned n=0; n<nd; ++n) {
        auto uv = u[n].view();
        const unsigned np = pmap->subCommSize(n);
        for (unsigned p=0; p<np; ++p) {
          auto sub = Kokkos::subview(
            u_overlapped[n].view(),
            std::make_pair(offsets[n][p], offsets[n][p]+sizes[n][p]), Kokkos::ALL);
          Kokkos::fence();
          if (pmap->subCommRank(n) == p)
            gt_assert(sub.span() == uv.span());
          pmap->subGridReduce(n, sub, uv, p);
        }
      }
    }
    else
      deep_copy(u, u_overlapped); // no-op if u and u_overlapped are the same
  }

  virtual void doExport(const KtensorT<ExecSpace>& u,
                        const KtensorT<ExecSpace>& u_overlapped,
                        const ttb_indx n) const override
  {
    GENTEN_TIME_MONITOR("k-tensor export");

    if (pmap != nullptr) {
      auto uv = u[n].view();
      const unsigned np = pmap->subCommSize(n);
      for (unsigned p=0; p<np; ++p) {
        auto sub = Kokkos::subview(
          u_overlapped[n].view(),
          std::make_pair(offsets[n][p], offsets[n][p]+sizes[n][p]), Kokkos::ALL);
        Kokkos::fence();
        if (pmap->subCommRank(n) == p && sub.span() != uv.span()) {
          Genten::error("Spans do not match!");
        }
        pmap->subGridReduce(n, sub, uv, p);
      }
    }
    else
      deep_copy(u[n], u_overlapped[n]); // no-op if u and u_overlapped are the same
  }

};

template <typename ExecSpace>
class KtensorOneSidedUpdate :
    public DistKtensorUpdate<ExecSpace> {
private:
  const ProcessorMap *pmap;
  bool parallel;
  std::vector< std::vector<int> > offsets;
  std::vector< std::vector<int> > sizes;

  std::vector< std::vector<int> > offsets_r;
  std::vector< std::vector<int> > sizes_r;

  bool sparse;
  SptensorT<ExecSpace> X_sparse;

public:
  using unordered_map_type =
    Kokkos::UnorderedMap<ttb_indx,unsigned,Kokkos::HostSpace>;
  std::vector<unordered_map_type> maps;

private:

#ifdef HAVE_DIST
  using umv_type = Kokkos::View<ttb_real**, Kokkos::LayoutStride, ExecSpace, Kokkos::MemoryUnmanaged>;
  std::vector<MPI_Win> windows;
  std::vector<umv_type> bufs;
#endif

public:
  KtensorOneSidedUpdate(const DistTensor<ExecSpace>& X,
                                       const KtensorT<ExecSpace>& u);
  virtual ~KtensorOneSidedUpdate();

  KtensorOneSidedUpdate(KtensorOneSidedUpdate&&) = default;
  KtensorOneSidedUpdate(const KtensorOneSidedUpdate&) = default;
  KtensorOneSidedUpdate& operator=(KtensorOneSidedUpdate&&) = default;
  KtensorOneSidedUpdate& operator=(const KtensorOneSidedUpdate&) = default;

  virtual void updateTensor(const DistTensor<ExecSpace>& X) override;

  virtual bool overlapDependsOnTensor() const override { return false; }

  virtual KtensorT<ExecSpace>
  createOverlapKtensor(const KtensorT<ExecSpace>& u) const override;

  virtual bool overlapAliasesArg() const override { return !parallel; }

  virtual bool isReplicated() const override { return false; }

  using DistKtensorUpdate<ExecSpace>::doImport;

  virtual void doImport(const KtensorT<ExecSpace>& u_overlapped,
                        const KtensorT<ExecSpace>& u) const override;

  virtual void doImport(const KtensorT<ExecSpace>& u_overlapped,
                        const KtensorT<ExecSpace>& u,
                        const ttb_indx n) const override;

  using DistKtensorUpdate<ExecSpace>::doExport;

  virtual void doExport(const KtensorT<ExecSpace>& u,
                        const KtensorT<ExecSpace>& u_overlapped) const override;

  virtual void doExport(const KtensorT<ExecSpace>& u,
                        const KtensorT<ExecSpace>& u_overlapped,
                        const ttb_indx n) const override;

  void copyToWindows(const KtensorT<ExecSpace>& u) const;
  void copyFromWindows(const KtensorT<ExecSpace>& u) const;
  void zeroOutWindows() const;
  void lockWindows() const;
  void unlockWindows() const;
  void fenceWindows() const;
  void importRow(const unsigned n, const ttb_indx row,
                 const KtensorT<ExecSpace>& u,
                 const KtensorT<ExecSpace>& u_overlap) const;
  void exportRow(const unsigned n, const ttb_indx row,
                 const ArrayT<ExecSpace>& grad,
                 const KtensorT<ExecSpace>& g) const;

  unsigned find_proc_for_row(unsigned n, unsigned row) const;

private:

  void doImportSparse(const KtensorT<ExecSpace>& u_overlapped,
                      const KtensorT<ExecSpace>& u) const;

  void doImportSparse(const KtensorT<ExecSpace>& u_overlapped,
                      const KtensorT<ExecSpace>& u,
                      const ttb_indx n) const;

  void doImportDense(const KtensorT<ExecSpace>& u_overlapped,
                     const KtensorT<ExecSpace>& u) const;

  void doImportDense(const KtensorT<ExecSpace>& u_overlapped,
                     const KtensorT<ExecSpace>& u,
                     const ttb_indx n) const;

  void doExportSparse(const KtensorT<ExecSpace>& u,
                      const KtensorT<ExecSpace>& u_overlapped) const;

  void doExportSparse(const KtensorT<ExecSpace>& u,
                      const KtensorT<ExecSpace>& u_overlapped,
                      const ttb_indx n) const;

  void doExportDense(const KtensorT<ExecSpace>& u,
                     const KtensorT<ExecSpace>& u_overlapped) const;

  void doExportDense(const KtensorT<ExecSpace>& u,
                     const KtensorT<ExecSpace>& u_overlapped,
                     const ttb_indx n) const;

};

template <typename ExecSpace>
class KtensorTwoSidedUpdate :
    public DistKtensorUpdate<ExecSpace> {
private:
  const ProcessorMap *pmap;
  bool parallel;
  AlgParams algParams;

  std::vector< std::vector<int> > offsets;
  std::vector< std::vector<int> > sizes;

  std::vector< std::vector<int> > offsets_r;
  std::vector< std::vector<int> > sizes_r;

  bool sparse;
  SptensorT<ExecSpace> X_sparse;
  unsigned nc;

  using offsets_type = Kokkos::View<int*,ExecSpace>;
  // Set the host execution space to be ExecSpace if it is a host execution
  // space.  This avoids using OpenMP instead of Serial when both are enabled
  // when doing:
  //   using host_offsets_type = Kokkos::View<int*,Kokkos::HostSpace>;
  //   using HostExecSpace = typename Kokkos::HostSpace::execution_space;
  using HostExecSpace =
    std::conditional_t<
      Kokkos::Impl::MemorySpaceAccess<
        Kokkos::HostSpace, typename ExecSpace::memory_space >::accessible,
      ExecSpace, typename Kokkos::HostSpace::execution_space >;
  using HostDevice = Kokkos::Device< HostExecSpace, Kokkos::HostSpace >;
  using host_offsets_type = Kokkos::View<int*,HostDevice>;
  std::vector< offsets_type > offsets_dev;

  // MPI_Alltoallv doesn't appear to work with these on the device
  // (for at least some MPI libraries)
  std::vector< host_offsets_type > num_row_sends;
  std::vector< host_offsets_type > num_row_recvs;
  std::vector< host_offsets_type > row_send_offsets;
  std::vector< host_offsets_type > row_recv_offsets;
  std::vector< host_offsets_type > num_fac_sends;
  std::vector< host_offsets_type > num_fac_recvs;
  std::vector< host_offsets_type > fac_send_offsets;
  std::vector< host_offsets_type > fac_recv_offsets;

  std::vector< offsets_type > num_row_recvs_dev;
  std::vector< offsets_type > num_fac_recvs_dev;
  std::vector< offsets_type > row_recv_offsets_dev;

  using row_vec_type = Kokkos::View<ttb_indx*,ExecSpace>;
  using fac_vec_type = Kokkos::View<ttb_real*,ExecSpace>;
  std::vector< row_vec_type > row_sends;
  std::vector< row_vec_type > row_recvs;
  std::vector< fac_vec_type > fac_sends;
  std::vector< fac_vec_type > fac_recvs;

  std::vector< std::unordered_map<ttb_indx, unsigned> > maps;
  std::vector< std::vector< std::vector<int> > > row_recvs_for_proc;

public:
  KtensorTwoSidedUpdate(const DistTensor<ExecSpace>& X,
                        const KtensorT<ExecSpace>& u,
                        const AlgParams& algParams);
  virtual ~KtensorTwoSidedUpdate();

  KtensorTwoSidedUpdate(KtensorTwoSidedUpdate&&) = default;
  KtensorTwoSidedUpdate(const KtensorTwoSidedUpdate&) = default;
  KtensorTwoSidedUpdate& operator=(KtensorTwoSidedUpdate&&) = default;
  KtensorTwoSidedUpdate& operator=(const KtensorTwoSidedUpdate&) = default;

  virtual void updateTensor(const DistTensor<ExecSpace>& X) override;

  virtual bool overlapDependsOnTensor() const override { return false; }

  virtual KtensorT<ExecSpace>
  createOverlapKtensor(const KtensorT<ExecSpace>& u) const override;

  virtual void initOverlapKtensor(KtensorT<ExecSpace>& u) const override;

  virtual bool overlapAliasesArg() const override { return !parallel; }

  virtual bool isReplicated() const override { return false; }

  using DistKtensorUpdate<ExecSpace>::doImport;

  virtual void doImport(const KtensorT<ExecSpace>& u_overlapped,
                        const KtensorT<ExecSpace>& u) const override;

  virtual void doImport(const KtensorT<ExecSpace>& u_overlapped,
                        const KtensorT<ExecSpace>& u,
                        const ttb_indx n) const override;

  using DistKtensorUpdate<ExecSpace>::doExport;

  virtual void doExport(const KtensorT<ExecSpace>& u,
                        const KtensorT<ExecSpace>& u_overlapped) const override;

  virtual void doExport(const KtensorT<ExecSpace>& u,
                        const KtensorT<ExecSpace>& u_overlapped,
                        const ttb_indx n) const override;

  // These have to be public for Cuda

  void extractRowRecvsDevice();

  void doImportSparse(const KtensorT<ExecSpace>& u_overlapped,
                      const KtensorT<ExecSpace>& u,
                      const ttb_indx n) const;

  void doExportSparse(const KtensorT<ExecSpace>& u,
                      const KtensorT<ExecSpace>& u_overlapped,
                      const ttb_indx n) const;

private:

  unsigned find_proc_for_row(unsigned n, unsigned row) const;

  void extractRowRecvsHost();

  void doImportSparse(const KtensorT<ExecSpace>& u_overlapped,
                      const KtensorT<ExecSpace>& u) const;

  void doImportDense(const KtensorT<ExecSpace>& u_overlapped,
                     const KtensorT<ExecSpace>& u) const;

  void doImportDense(const KtensorT<ExecSpace>& u_overlapped,
                     const KtensorT<ExecSpace>& u,
                     const ttb_indx n) const;

  void doExportSparse(const KtensorT<ExecSpace>& u,
                      const KtensorT<ExecSpace>& u_overlapped) const;

  void doExportDense(const KtensorT<ExecSpace>& u,
                     const KtensorT<ExecSpace>& u_overlapped) const;

  void doExportDense(const KtensorT<ExecSpace>& u,
                     const KtensorT<ExecSpace>& u_overlapped,
                     const ttb_indx n) const;

};

template <typename TensorType>
DistKtensorUpdate<typename TensorType::exec_space>*
createKtensorUpdate(const TensorType& X,
                    const KtensorT<typename TensorType::exec_space>& u,
                    const AlgParams& algParams)
{
  using exec_space = typename TensorType::exec_space;
  DistKtensorUpdate<exec_space>* dku = nullptr;
  if (algParams.dist_update_method == Dist_Update_Method::AllReduce)
    dku = new KtensorAllReduceUpdate<exec_space>(u);
#ifdef HAVE_TPETRA
  else if (algParams.dist_update_method == Dist_Update_Method::Tpetra)
    dku = new KtensorTpetraUpdate<exec_space>(X, u);
#endif
   else if (algParams.dist_update_method == Dist_Update_Method::AllGatherReduce)
    dku = new KtensorAllGatherReduceUpdate<exec_space>(u);
  else if (algParams.dist_update_method == Dist_Update_Method::OneSided)
    dku = new KtensorOneSidedUpdate<exec_space>(X, u);
   else if (algParams.dist_update_method == Dist_Update_Method::TwoSided)
     dku = new KtensorTwoSidedUpdate<exec_space>(X, u, algParams);
  else
    Genten::error("Unknown distributed Ktensor update method");
  return dku;
}

}
