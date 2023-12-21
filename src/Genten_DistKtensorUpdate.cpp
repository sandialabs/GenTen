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

#include "Genten_DistKtensorUpdate.hpp"
#include "Genten_Util.hpp"

namespace Genten {

template <typename ExecSpace>
KtensorOneSidedUpdate<ExecSpace>::
KtensorOneSidedUpdate(const DistTensor<ExecSpace>& X,
                      const KtensorT<ExecSpace>& u) :
  pmap(u.getProcessorMap())
{
  parallel = pmap != nullptr && pmap->gridSize() > 1;
  if (parallel) {
    const unsigned nd = u.ndims();
    sizes.resize(nd);
    sizes_r.resize(nd);
    offsets.resize(nd);
    offsets_r.resize(nd);
    for (unsigned n=0; n<nd; ++n) {
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
  }

  updateTensor(X); // build parallel RMA maps

#ifdef HAVE_DIST
  if (parallel) {
    const unsigned nd = u.ndims();
    windows.resize(nd);
    bufs.resize(nd);
    for (unsigned n=0; n<nd; ++n) {
      const ttb_indx nrow   = u[n].view().extent(0);
      const ttb_indx ncol   = u[n].view().extent(1);
      const ttb_indx stride = u[n].view().stride(0);
      const MPI_Aint sz     = u[n].view().span()*sizeof(ttb_real);
      ttb_real *buf;
      MPI_Win_allocate(sz, int(sizeof(ttb_real)), MPI_INFO_NULL,
                       pmap->subComm(n), &buf, &windows[n]);
      Kokkos::LayoutStride layout(nrow, stride, ncol, 1);
      bufs[n] = umv_type(buf, layout);
    }
  }
#endif
}

template <typename ExecSpace>
KtensorOneSidedUpdate<ExecSpace>::
~KtensorOneSidedUpdate()
{
#ifdef HAVE_DIST
  if (parallel) {
    unsigned nd = windows.size();
    for (unsigned n=0; n<nd; ++n)
      MPI_Win_free(&windows[n]);
  }
#endif
}

template <typename ExecSpace>
void
KtensorOneSidedUpdate<ExecSpace>::
updateTensor(const DistTensor<ExecSpace>& X)
{
  GENTEN_TIME_MONITOR("update tensor");
  sparse = X.isSparse();

  if (sparse && parallel) {
    X_sparse = X.getSptensor();
    const ttb_indx nnz = X_sparse.nnz();
    const unsigned nd = X_sparse.ndims();
    maps.resize(nd);
    for (unsigned n=0; n<nd; ++n) {
      const unsigned np = offsets[n].size();
      const ttb_indx nrows = offsets[n][np-1]+sizes[n][np-1];
      maps[n] = unordered_map_type(nrows);
    }

    auto X_sparse_h = create_mirror_view(X_sparse);
    deep_copy(X_sparse_h, X_sparse);
    for (ttb_indx i=0; i<nnz; ++i) {
      const auto subs = X_sparse_h.getSubscripts(i);
      for (unsigned n=0; n<nd; ++n) {
        const ttb_indx row = subs[n];
        if (!maps[n].exists(row)) {
          // Find processor given row lives on by finding processor where
          // the first row is bigger than the given row, then move back one
          const unsigned np = pmap->subCommSize(n);
          unsigned p = 0;
          while (p < np && row >= ttb_indx(offsets[n][p])) ++p;
          --p;
          gt_assert(!maps[n].insert(row,p).failed());
        }
      }
    }
  }
}

template <typename ExecSpace>
KtensorT<ExecSpace>
KtensorOneSidedUpdate<ExecSpace>::
createOverlapKtensor(const KtensorT<ExecSpace>& u) const
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
}

template <typename ExecSpace>
void
KtensorOneSidedUpdate<ExecSpace>::
doImport(const KtensorT<ExecSpace>& u_overlapped,
         const KtensorT<ExecSpace>& u) const
{
  GENTEN_TIME_MONITOR("k-tensor import");
  if (parallel) {
    if (sparse)
      doImportSparse(u_overlapped, u);
    else
      doImportDense(u_overlapped, u);
  }
  else
    deep_copy(u_overlapped, u); // no-op if u and u_overlapped are the same
}

template <typename ExecSpace>
void
KtensorOneSidedUpdate<ExecSpace>::
doImport(const KtensorT<ExecSpace>& u_overlapped,
         const KtensorT<ExecSpace>& u,
         const ttb_indx n) const
{
  GENTEN_TIME_MONITOR("k-tensor import");
  if (parallel) {
    if (sparse)
      doImportSparse(u_overlapped, u, n);
    else
      doImportDense(u_overlapped, u, n);
  }
  else
    deep_copy(u_overlapped[n], u[n]); // no-op if u and u_overlapped are the same
}

template <typename ExecSpace>
void
KtensorOneSidedUpdate<ExecSpace>::
doExport(const KtensorT<ExecSpace>& u,
         const KtensorT<ExecSpace>& u_overlapped) const
{
  GENTEN_TIME_MONITOR("k-tensor export");
  if (parallel) {
    if (sparse)
      doExportSparse(u, u_overlapped);
    else
      doExportDense(u, u_overlapped);
  }
  else
    deep_copy(u, u_overlapped); // no-op if u and u_overlapped are the same
}

template <typename ExecSpace>
void
KtensorOneSidedUpdate<ExecSpace>::
doExport(const KtensorT<ExecSpace>& u,
         const KtensorT<ExecSpace>& u_overlapped,
         const ttb_indx n) const
{
  GENTEN_TIME_MONITOR("k-tensor export");
  if (parallel) {
    if (sparse)
      doExportSparse(u, u_overlapped, n);
    else
      doExportDense(u, u_overlapped, n);
  }
  else
    deep_copy(u[n], u_overlapped[n]); // no-op if u and u_overlapped are the same
}

template <typename ExecSpace>
void
KtensorOneSidedUpdate<ExecSpace>::
doImportSparse(const KtensorT<ExecSpace>& u_overlapped,
               const KtensorT<ExecSpace>& u) const
{
#ifdef HAVE_DIST
  const unsigned nd = u.ndims();
  for (unsigned n=0; n<nd; ++n) {
    const unsigned rank = pmap->subCommRank(n);
    const unsigned np = pmap->subCommSize(n);
    gt_assert(u[n].view().span() == size_t(sizes_r[n][rank]));
    gt_assert(u_overlapped[n].view().span() == size_t(offsets_r[n][np-1]+sizes_r[n][np-1]));

    // Copy u into our window
    MPI_Win_fence(0, windows[n]);
    Kokkos::deep_copy(bufs[n], u[n].view());
    MPI_Win_fence(0, windows[n]);
  }

  for (unsigned n=0; n<nd; ++n)
    MPI_Win_lock_all(0, windows[n]);

  // Only fill u_overlapped[n] for the rows corresponding to nonzeros in X
  for (unsigned n=0; n<nd; ++n) {
    const ttb_indx nrow = maps[n].capacity();
    const ttb_indx stride_u = u_overlapped[n].view().stride(0);
    const ttb_indx stride_b = bufs[n].stride(0);
    for (ttb_indx i=0; i<nrow; ++i) {
      if (maps[n].valid_at(i)) {
        const ttb_indx row = maps[n].key_at(i);
        const unsigned p = maps[n].value_at(i);

        // Grab the given row from the given processor
        ttb_real *ptr = u_overlapped[n].view().data()+row*stride_u;
        const int cnt = u.ncomponents();
        const MPI_Aint beg = (row-offsets[n][p])*stride_b;
        gt_assert(int(beg) < sizes_r[n][p]);
        gt_assert(int(beg+cnt) <= sizes_r[n][p]);
        MPI_Get(ptr, cnt, DistContext::toMpiType<ttb_real>(), p,
                beg, cnt, DistContext::toMpiType<ttb_real>(), windows[n]);
      }
    }
  }

  for (unsigned n=0; n<nd; ++n)
    MPI_Win_unlock_all(windows[n]);
#endif
}

template <typename ExecSpace>
void
KtensorOneSidedUpdate<ExecSpace>::
doImportSparse(const KtensorT<ExecSpace>& u_overlapped,
               const KtensorT<ExecSpace>& u,
               const ttb_indx n) const
{
#ifdef HAVE_DIST
  const unsigned rank = pmap->subCommRank(n);
  const unsigned np = pmap->subCommSize(n);
  gt_assert(u[n].view().span() == size_t(sizes_r[n][rank]));
  gt_assert(u_overlapped[n].view().span() == size_t(offsets_r[n][np-1]+sizes_r[n][np-1]));

  // Copy u into our window
  MPI_Win_fence(0, windows[n]);
  Kokkos::deep_copy(bufs[n], u[n].view());
  MPI_Win_fence(0, windows[n]);

  // Only fill u_overlapped[n] for the rows corresponding to nonzeros in X
  MPI_Win_lock_all(0, windows[n]);
  const ttb_indx nrow = maps[n].capacity();
  const ttb_indx stride_u = u_overlapped[n].view().stride(0);
  const ttb_indx stride_b = bufs[n].stride(0);
  for (ttb_indx i=0; i<nrow; ++i) {
    if (maps[n].valid_at(i)) {
      const ttb_indx row = maps[n].key_at(i);
      const unsigned p = maps[n].value_at(i);

      // Grab the given row from the given processor
      ttb_real *ptr = u_overlapped[n].view().data()+row*stride_u;
      const int cnt = u.ncomponents();
      const MPI_Aint beg = (row-offsets[n][p])*stride_b;
      gt_assert(int(beg) < sizes_r[n][p]);
      gt_assert(int(beg+cnt) <= sizes_r[n][p]);
      MPI_Get(ptr, cnt, DistContext::toMpiType<ttb_real>(), p,
              beg, cnt, DistContext::toMpiType<ttb_real>(), windows[n]);
    }
  }
  MPI_Win_unlock_all(windows[n]);
#endif
}

template <typename ExecSpace>
void
KtensorOneSidedUpdate<ExecSpace>::
doImportDense(const KtensorT<ExecSpace>& u_overlapped,
              const KtensorT<ExecSpace>& u) const
{
#ifdef HAVE_DIST
  const unsigned nd = u.ndims();
  for (unsigned n=0; n<nd; ++n) {
    const unsigned rank = pmap->subCommRank(n);
    const unsigned np = pmap->subCommSize(n);
    gt_assert(u[n].view().span() == size_t(sizes_r[n][rank]));
    gt_assert(u_overlapped[n].view().span() == size_t(offsets_r[n][np-1]+sizes_r[n][np-1]));

    // Copy u into our window
    MPI_Win_fence(0, windows[n]);
    Kokkos::deep_copy(bufs[n], u[n].view());
    MPI_Win_fence(0, windows[n]);
  }

  for (unsigned n=0; n<nd; ++n)
    MPI_Win_lock_all(0, windows[n]);

  // Get data from all ranks
  for (unsigned n=0; n<nd; ++n) {
    const unsigned np = pmap->subCommSize(n);
    for (unsigned p=0; p<np; ++p) {
      ttb_real *ptr = u_overlapped[n].view().data()+offsets_r[n][p];
      const int cnt = sizes_r[n][p];
      const MPI_Aint beg = 0;
      MPI_Get(ptr, cnt, DistContext::toMpiType<ttb_real>(), p,
              beg, cnt, DistContext::toMpiType<ttb_real>(), windows[n]);
    }
  }

  for (unsigned n=0; n<nd; ++n)
    MPI_Win_unlock_all(windows[n]);
#endif
}

template <typename ExecSpace>
void
KtensorOneSidedUpdate<ExecSpace>::
doImportDense(const KtensorT<ExecSpace>& u_overlapped,
              const KtensorT<ExecSpace>& u,
              const ttb_indx n) const
{
#ifdef HAVE_DIST
  const unsigned rank = pmap->subCommRank(n);
  const unsigned np = pmap->subCommSize(n);
  gt_assert(u[n].view().span() == size_t(sizes_r[n][rank]));
  gt_assert(u_overlapped[n].view().span() == size_t(offsets_r[n][np-1]+sizes_r[n][np-1]));

  // Copy u into our window
  MPI_Win_fence(0, windows[n]);
  Kokkos::deep_copy(bufs[n], u[n].view());
  MPI_Win_fence(0, windows[n]);

  // Get data from all ranks
  MPI_Win_lock_all(0, windows[n]);
  for (unsigned p=0; p<np; ++p) {
    ttb_real *ptr = u_overlapped[n].view().data()+offsets_r[n][p];
    const int cnt = sizes_r[n][p];
    const MPI_Aint beg = 0;
    MPI_Get(ptr, cnt, DistContext::toMpiType<ttb_real>(), p,
            beg, cnt, DistContext::toMpiType<ttb_real>(), windows[n]);
  }
  MPI_Win_unlock_all(windows[n]);
#endif
}

template <typename ExecSpace>
void
KtensorOneSidedUpdate<ExecSpace>::
doExportSparse(const KtensorT<ExecSpace>& u,
               const KtensorT<ExecSpace>& u_overlapped) const
{
#ifdef HAVE_DIST
  const unsigned nd = u.ndims();
  for (unsigned n=0; n<nd; ++n) {
    const unsigned rank = pmap->subCommRank(n);
    const unsigned np = pmap->subCommSize(n);
    gt_assert(u[n].view().span() == size_t(sizes_r[n][rank]));
    gt_assert(u_overlapped[n].view().span() == size_t(offsets_r[n][np-1]+sizes_r[n][np-1]));

    // Zero out window
    MPI_Win_fence(0, windows[n]);
    Kokkos::deep_copy(bufs[n], ttb_real(0.0));
    MPI_Win_fence(0, windows[n]);
  }

  for (unsigned n=0; n<nd; ++n)
    MPI_Win_lock_all(0, windows[n]);

  // Only accumulate u[n] for the rows corespondong to nonzeros in X
  for (unsigned n=0; n<nd; ++n) {
    const ttb_indx nrow = maps[n].capacity();
    const ttb_indx stride_u = u_overlapped[n].view().stride(0);
    const ttb_indx stride_b = bufs[n].stride(0);
    for (ttb_indx i=0; i<nrow; ++i) {
      if (maps[n].valid_at(i)) {
        const ttb_indx row = maps[n].key_at(i);
        const unsigned p = maps[n].value_at(i);
        const ttb_real *ptr = u_overlapped[n].view().data()+row*stride_u;
        const int cnt = u.ncomponents();
        const MPI_Aint beg = (row-offsets[n][p])*stride_b;
        gt_assert(int(beg) < sizes_r[n][p]);
        gt_assert(int(beg+cnt) <= sizes_r[n][p]);
        MPI_Accumulate(ptr, cnt, DistContext::toMpiType<ttb_real>(), p,
                       beg, cnt, DistContext::toMpiType<ttb_real>(), MPI_SUM,
                       windows[n]);
      }
    }
  }

  for (unsigned n=0; n<nd; ++n)
    MPI_Win_unlock_all(windows[n]);

  // Copy window data into u
  for (unsigned n=0; n<nd; ++n) {
    MPI_Win_fence(0, windows[n]);
    Kokkos::deep_copy(u[n].view(), bufs[n]);
  }
#endif
}

template <typename ExecSpace>
void
KtensorOneSidedUpdate<ExecSpace>::
doExportSparse(const KtensorT<ExecSpace>& u,
               const KtensorT<ExecSpace>& u_overlapped,
               const ttb_indx n) const
{
#ifdef HAVE_DIST
  const unsigned rank = pmap->subCommRank(n);
  const unsigned np = pmap->subCommSize(n);
  gt_assert(u[n].view().span() == size_t(sizes_r[n][rank]));
  gt_assert(u_overlapped[n].view().span() == size_t(offsets_r[n][np-1]+sizes_r[n][np-1]));

  // Zero out window
  MPI_Win_fence(0, windows[n]);
  Kokkos::deep_copy(bufs[n], ttb_real(0.0));
  MPI_Win_fence(0, windows[n]);

  // Only accumulate u[n] for the rows corespondong to nonzeros in X
  MPI_Win_lock_all(0, windows[n]);
  const ttb_indx nrow = maps[n].capacity();
  const ttb_indx stride_u = u_overlapped[n].view().stride(0);
  const ttb_indx stride_b = bufs[n].stride(0);
  for (ttb_indx i=0; i<nrow; ++i) {
    if (maps[n].valid_at(i)) {
      const ttb_indx row = maps[n].key_at(i);
      const unsigned p = maps[n].value_at(i);
      const ttb_real *ptr = u_overlapped[n].view().data()+row*stride_u;
      const int cnt = u.ncomponents();
      const MPI_Aint beg = (row-offsets[n][p])*stride_b;
      gt_assert(int(beg) < sizes_r[n][p]);
      gt_assert(int(beg+cnt) <= sizes_r[n][p]);
      MPI_Accumulate(ptr, cnt, DistContext::toMpiType<ttb_real>(), p,
                     beg, cnt, DistContext::toMpiType<ttb_real>(), MPI_SUM,
                     windows[n]);
    }
  }
  MPI_Win_unlock_all(windows[n]);

  // Copy window data into u
  MPI_Win_fence(0, windows[n]);
  Kokkos::deep_copy(u[n].view(), bufs[n]);
#endif
}

template <typename ExecSpace>
void
KtensorOneSidedUpdate<ExecSpace>::
doExportDense(const KtensorT<ExecSpace>& u,
              const KtensorT<ExecSpace>& u_overlapped) const
{
#ifdef HAVE_DIST
  const unsigned nd = u.ndims();
  for (unsigned n=0; n<nd; ++n) {
    const unsigned rank = pmap->subCommRank(n);
    const unsigned np = pmap->subCommSize(n);
    gt_assert(u[n].view().span() == size_t(sizes_r[n][rank]));
    gt_assert(u_overlapped[n].view().span() == size_t(offsets_r[n][np-1]+sizes_r[n][np-1]));

    // Zero out window
    MPI_Win_fence(0, windows[n]);
    Kokkos::deep_copy(bufs[n], ttb_real(0.0));
    MPI_Win_fence(0, windows[n]);
  }

  for (unsigned n=0; n<nd; ++n)
    MPI_Win_lock_all(0, windows[n]);

  // Accumulate data to all ranks
  for (unsigned n=0; n<nd; ++n) {
    const unsigned np = pmap->subCommSize(n);
    for (unsigned p=0; p<np; ++p) {
      const ttb_real *ptr = u_overlapped[n].view().data()+offsets_r[n][p];
      const int cnt = sizes_r[n][p];
      const MPI_Aint beg = 0;
      MPI_Accumulate(ptr, cnt, DistContext::toMpiType<ttb_real>(), p,
                     beg, cnt, DistContext::toMpiType<ttb_real>(), MPI_SUM,
                     windows[n]);
    }
  }

  for (unsigned n=0; n<nd; ++n)
    MPI_Win_unlock_all(windows[n]);

  // Copy window data into u
  for (unsigned n=0; n<nd; ++n) {
    MPI_Win_fence(0, windows[n]);
    Kokkos::deep_copy(u[n].view(), bufs[n]);
  }
#endif
}

template <typename ExecSpace>
void
KtensorOneSidedUpdate<ExecSpace>::
doExportDense(const KtensorT<ExecSpace>& u,
              const KtensorT<ExecSpace>& u_overlapped,
              const ttb_indx n) const
{
#ifdef HAVE_DIST
  const unsigned rank = pmap->subCommRank(n);
  const unsigned np = pmap->subCommSize(n);
  gt_assert(u[n].view().span() == size_t(sizes_r[n][rank]));
  gt_assert(u_overlapped[n].view().span() == size_t(offsets_r[n][np-1]+sizes_r[n][np-1]));

  // Zero out window
  MPI_Win_fence(0, windows[n]);
  Kokkos::deep_copy(bufs[n], ttb_real(0.0));
  MPI_Win_fence(0, windows[n]);

  // Accumulate data to all ranks
  MPI_Win_lock_all(0, windows[n]);
  for (unsigned p=0; p<np; ++p) {
    const ttb_real *ptr = u_overlapped[n].view().data()+offsets_r[n][p];
    const int cnt = sizes_r[n][p];
    const MPI_Aint beg = 0;
    MPI_Accumulate(ptr, cnt, DistContext::toMpiType<ttb_real>(), p,
                   beg, cnt, DistContext::toMpiType<ttb_real>(), MPI_SUM,
                   windows[n]);
  }
  MPI_Win_unlock_all(windows[n]);

  // Copy window data into u
  MPI_Win_fence(0, windows[n]);
  Kokkos::deep_copy(u[n].view(), bufs[n]);
#endif
}

}

#define INST_MACRO(SPACE) template class Genten::KtensorOneSidedUpdate<SPACE>;
GENTEN_INST(INST_MACRO)
