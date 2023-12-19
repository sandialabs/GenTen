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

  virtual void updateTensor(DistTensor<ExecSpace>& X) {}

  virtual KtensorT<ExecSpace>
  createOverlapKtensor(const KtensorT<ExecSpace>& u) const
  {
    return u;
  };

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

  virtual void updateTensor(DistTensor<ExecSpace>& X_) override
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
class KtensorOneSidedAllGatherReduceUpdate :
    public DistKtensorUpdate<ExecSpace> {
private:
  const ProcessorMap *pmap;
  bool parallel;
  std::vector< std::vector<int> > offsets;
  std::vector< std::vector<int> > sizes;

  std::vector< std::vector<int> > offsets_r;
  std::vector< std::vector<int> > sizes_r;

#ifdef HAVE_DIST
  using umv_type = Kokkos::View<ttb_real**, Kokkos::LayoutRight, ExecSpace, Kokkos::MemoryUnmanaged>;
  std::vector<MPI_Win> windows;
  std::vector<umv_type> bufs;
#endif

public:
  KtensorOneSidedAllGatherReduceUpdate(const KtensorT<ExecSpace>& u) :
    pmap(u.getProcessorMap())
  {
    parallel = pmap != nullptr && pmap->gridSize() > 1;
    if (!parallel)
      return;

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

#ifdef HAVE_DIST
    windows.resize(nd);
    bufs.resize(nd);
    for (unsigned n=0; n<nd; ++n) {
      const unsigned np = offsets[n].size();
      const ttb_indx nr = offsets[n][np-1]+sizes[n][np-1];
      const ttb_indx nc = (offsets_r[n][np-1]+sizes_r[n][np-1])/nr;
      const MPI_Aint sz = nr*nc*sizeof(ttb_real);
      ttb_real *buf;
      MPI_Win_allocate(sz, int(sizeof(ttb_real)), MPI_INFO_NULL,
                       pmap->subComm(n), &buf, &windows[n]);
      bufs[n] = umv_type(buf, nr, nc);
    }
#endif
  }
  virtual ~KtensorOneSidedAllGatherReduceUpdate()
  {
#ifdef HAVE_DIST
    unsigned nd = windows.size();
    for (unsigned n=0; n<nd; ++n)
      MPI_Win_free(&windows[n]);
#endif
  }

  KtensorOneSidedAllGatherReduceUpdate(KtensorOneSidedAllGatherReduceUpdate&&) = default;
  KtensorOneSidedAllGatherReduceUpdate(const KtensorOneSidedAllGatherReduceUpdate&) = default;
  KtensorOneSidedAllGatherReduceUpdate& operator=(KtensorOneSidedAllGatherReduceUpdate&&) = default;
  KtensorOneSidedAllGatherReduceUpdate& operator=(const KtensorOneSidedAllGatherReduceUpdate&) = default;

  virtual KtensorT<ExecSpace>
  createOverlapKtensor(const KtensorT<ExecSpace>& u) const override
  {
    GENTEN_TIME_MONITOR("create overlapped k-tensor");
    if (!parallel)
      return u;

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

  virtual bool overlapAliasesArg() const override { return !parallel; }

  virtual bool isReplicated() const override { return false; }

  using DistKtensorUpdate<ExecSpace>::doImport;

  virtual void doImport(const KtensorT<ExecSpace>& u_overlapped,
                        const KtensorT<ExecSpace>& u) const override
  {
    const unsigned nd = u.ndims();
    for (unsigned n=0; n<nd; ++n)
      this->doImport(u_overlapped, u, n);
  }

  virtual void doImport(const KtensorT<ExecSpace>& u_overlapped,
                        const KtensorT<ExecSpace>& u,
                        const ttb_indx n) const override
  {
    GENTEN_TIME_MONITOR("k-tensor import");

#ifdef HAVE_DIST
    if (parallel) {
      const unsigned rank = pmap->subCommRank(n);
      const unsigned np = pmap->subCommSize(n);
      gt_assert(u[n].view().span() == size_t(sizes_r[n][rank]));
      gt_assert(u_overlapped[n].view().span() == size_t(offsets_r[n][np-1]+sizes_r[n][np-1]));

      MPI_Win_fence(0, windows[n]);

      // Copy u into our window
      const int my_beg = offsets[n][rank];
      const int my_end = offsets[n][rank]+sizes[n][rank];
      const unsigned nc = u.ncomponents();
      auto sub_buf = Kokkos::subview(bufs[n], std::make_pair(my_beg, my_end),
                                     std::make_pair(0u,nc));
      Kokkos::deep_copy(sub_buf, u[n].view());

      Kokkos::fence();
      MPI_Win_fence(0, windows[n]);

      // Get data from all ranks
      for (unsigned p=0; p<np; ++p) {
        ttb_real *ptr = u_overlapped[n].view().data()+offsets_r[n][p];
        const int cnt = sizes_r[n][p];
        const MPI_Aint beg = offsets_r[n][p];
        MPI_Get(ptr, cnt, DistContext::toMpiType<ttb_real>(), p,
                beg, cnt, DistContext::toMpiType<ttb_real>(), windows[n]);
      }

      MPI_Win_fence(0, windows[n]);
    }
    else
      deep_copy(u_overlapped[n], u[n]); // no-op if u and u_overlapped are the same
#else
    deep_copy(u_overlapped[n], u[n]); // no-op if u and u_overlapped are the same
#endif
  }

  using DistKtensorUpdate<ExecSpace>::doExport;

  virtual void doExport(const KtensorT<ExecSpace>& u,
                        const KtensorT<ExecSpace>& u_overlapped) const override
  {
    const unsigned nd = u.ndims();
    for (unsigned n=0; n<nd; ++n)
      this->doExport(u, u_overlapped, n);
  }

  virtual void doExport(const KtensorT<ExecSpace>& u,
                        const KtensorT<ExecSpace>& u_overlapped,
                        const ttb_indx n) const override
  {
    GENTEN_TIME_MONITOR("k-tensor export");

#ifdef HAVE_DIST
    if (parallel) {
      const unsigned rank = pmap->subCommRank(n);
      const unsigned np = pmap->subCommSize(n);
      gt_assert(u[n].view().span() == size_t(sizes_r[n][rank]));
      gt_assert(u_overlapped[n].view().span() == size_t(offsets_r[n][np-1]+sizes_r[n][np-1]));

      MPI_Win_fence(0, windows[n]);

      // Zero out window
      Kokkos::deep_copy(bufs[n], ttb_real(0.0));
      Kokkos::fence();
      MPI_Win_fence(0, windows[n]);

      // Accumulate data to all ranks
      for (unsigned p=0; p<np; ++p) {
        const ttb_real *ptr = u_overlapped[n].view().data()+offsets_r[n][p];
        const int cnt = sizes_r[n][p];
        const MPI_Aint beg = offsets_r[n][p];
        MPI_Accumulate(ptr, cnt, DistContext::toMpiType<ttb_real>(), p,
                       beg, cnt, DistContext::toMpiType<ttb_real>(), MPI_SUM,
                       windows[n]);
      }

      MPI_Win_fence(0, windows[n]);

      // Copy window data into u
      const int my_beg = offsets[n][rank];
      const int my_end = offsets[n][rank]+sizes[n][rank];
      const unsigned nc = u.ncomponents();
      auto sub_buf = Kokkos::subview(bufs[n], std::make_pair(my_beg, my_end),
                                     std::make_pair(0u,nc));
      Kokkos::deep_copy(u[n].view(), sub_buf);
    }
    else
      deep_copy(u[n], u_overlapped[n]); // no-op if u and u_overlapped are the same
#else
    deep_copy(u[n], u_overlapped[n]); // no-op if u and u_overlapped are the same
#endif
  }

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
  else if (algParams.dist_update_method == Dist_Update_Method::OneSidedAllGatherReduce)
    dku = new KtensorOneSidedAllGatherReduceUpdate<exec_space>(u);
  else
    Genten::error("Unknown distributed Ktensor update method");
  return dku;
}

template <typename TensorType>
DistKtensorUpdate<typename TensorType::exec_space>*
createKtensorUpdate(const TensorType& X,
                    const KtensorT<typename TensorType::exec_space>& u,
                    const ttb_indx nnz,
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
  else if (algParams.dist_update_method == Dist_Update_Method::OneSidedAllGatherReduce)
    dku = new KtensorOneSidedAllGatherReduceUpdate<exec_space>(u);
  else
    Genten::error("Unknown distributed Ktensor update method");
  return dku;
}

}
