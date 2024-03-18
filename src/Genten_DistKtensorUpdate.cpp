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
  parallel = pmap == nullptr || (pmap != nullptr && pmap->gridSize() > 1);
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
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "no_locks", "true");
    MPI_Info_set(info, "accumulate_ordering", "none");
    MPI_Info_set(info, "accumulate_ops", "same_op");
    for (unsigned n=0; n<nd; ++n) {
      const ttb_indx nrow   = u[n].view().extent(0);
      const ttb_indx ncol   = u[n].view().extent(1);
      const ttb_indx stride = u[n].view().stride(0);
      const MPI_Aint sz     = u[n].view().span()*sizeof(ttb_real);
      ttb_real *buf;
      MPI_Win_allocate(sz, int(sizeof(ttb_real)), info,
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
          unsigned p = find_proc_for_row(n, row);
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
copyToWindows(const KtensorT<ExecSpace>& u) const
{
#ifdef HAVE_DIST
  if (parallel) {
    const unsigned nd = u.ndims();
    for (unsigned n=0; n<nd; ++n) {
      Kokkos::deep_copy(bufs[n], u[n].view());
    }
  }
#endif
}

template <typename ExecSpace>
void
KtensorOneSidedUpdate<ExecSpace>::
copyFromWindows(const KtensorT<ExecSpace>& u) const
{
#ifdef HAVE_DIST
  if (parallel) {
    const unsigned nd = u.ndims();
    for (unsigned n=0; n<nd; ++n) {
      Kokkos::deep_copy(u[n].view(), bufs[n]);
    }
  }
#endif
}

template <typename ExecSpace>
void
KtensorOneSidedUpdate<ExecSpace>::
zeroOutWindows() const
{
#ifdef HAVE_DIST
  if (parallel) {
    const unsigned nd = windows.size();
    for (unsigned n=0; n<nd; ++n) {
      Kokkos::deep_copy(bufs[n], ttb_real(0.0));
    }
  }
#endif
}

template <typename ExecSpace>
void
KtensorOneSidedUpdate<ExecSpace>::
lockWindows() const
{
#ifdef HAVE_DIST
  if (parallel) {
    const unsigned nd = windows.size();
    for (unsigned n=0; n<nd; ++n)
       MPI_Win_fence(MPI_MODE_NOPRECEDE, windows[n]);
  }
#endif
}

template <typename ExecSpace>
void
KtensorOneSidedUpdate<ExecSpace>::
unlockWindows() const
{
#ifdef HAVE_DIST
  if (parallel) {
    const unsigned nd = windows.size();
    for (unsigned n=0; n<nd; ++n)
      MPI_Win_fence(MPI_MODE_NOPUT+MPI_MODE_NOSTORE+MPI_MODE_NOSUCCEED, windows[n]);
  }
#endif
}

template <typename ExecSpace>
void
KtensorOneSidedUpdate<ExecSpace>::
fenceWindows() const
{
#ifdef HAVE_DIST
  if (parallel) {
    const unsigned nd = windows.size();
    for (unsigned n=0; n<nd; ++n)
      MPI_Win_fence(0, windows[n]);
  }
#endif
}

template <typename ExecSpace>
void
KtensorOneSidedUpdate<ExecSpace>::
importRow(const unsigned n, const ttb_indx row, const KtensorT<ExecSpace>& u,
          const KtensorT<ExecSpace>& u_overlap) const
{
#ifdef HAVE_DIST
  if (parallel) {
    const unsigned rank = pmap->subCommRank(n);
    const unsigned np = pmap->subCommSize(n);
    gt_assert(u[n].view().span() == size_t(sizes_r[n][rank]));
    gt_assert(u_overlap[n].view().span() == size_t(offsets_r[n][np-1]+sizes_r[n][np-1]));
    const ttb_indx stride_u = u_overlap[n].view().stride(0);
    const ttb_indx stride_b = bufs[n].stride(0);
    const unsigned p = find_proc_for_row(n, row);

    // Grab the given row from the given processor
    ttb_real *ptr = u_overlap[n].view().data()+row*stride_u;
    const int cnt = u_overlap.ncomponents();
    const MPI_Aint beg = (row-offsets[n][p])*stride_b;
    gt_assert(int(beg) < sizes_r[n][p]);
    gt_assert(int(beg+cnt) <= sizes_r[n][p]);
    MPI_Get(ptr, cnt, DistContext::toMpiType<ttb_real>(), p,
            beg, cnt, DistContext::toMpiType<ttb_real>(), windows[n]);
  }
  else
#endif
    if (u[n].view().data() != u_overlap[n].view().data())

    Kokkos::deep_copy(Kokkos::subview(u_overlap[n].view(), row, Kokkos::ALL),
                      Kokkos::subview(u[n].view(), row, Kokkos::ALL));
}

template <typename ExecSpace>
void
KtensorOneSidedUpdate<ExecSpace>::
exportRow(const unsigned n, const ttb_indx row, const ArrayT<ExecSpace>& grad,
          const KtensorT<ExecSpace>& g) const
{
#ifdef HAVE_DIST
  if (parallel) {
    const ttb_indx stride_b = bufs[n].stride(0);
    const unsigned p = find_proc_for_row(n, row);

    // Accumulate into the given row on the given processor
    const ttb_real *ptr = grad.ptr();
    const int cnt = grad.size();
    const MPI_Aint beg = (row-offsets[n][p])*stride_b;
    gt_assert(int(beg) < sizes_r[n][p]);
    gt_assert(int(beg+cnt) <= sizes_r[n][p]);
    MPI_Accumulate(ptr, cnt, DistContext::toMpiType<ttb_real>(), p,
                   beg, cnt, DistContext::toMpiType<ttb_real>(), MPI_SUM,
                   windows[n]);
  }
  else
#endif
    Kokkos::deep_copy(Kokkos::subview(g[n].view(), row, Kokkos::ALL),
                      grad.values());
}

template <typename ExecSpace>
unsigned
KtensorOneSidedUpdate<ExecSpace>::
find_proc_for_row(unsigned n, unsigned row) const
{
  // Find processor given row lives on by finding processor where
  // the first row is bigger than the given row, then move back one
  const unsigned np = pmap->subCommSize(n);
  unsigned p = 0;
  while (p < np && row >= ttb_indx(offsets[n][p])) ++p;
  --p;
  return p;
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
    Kokkos::deep_copy(bufs[n], u[n].view());

    // Tell MPI we are going to start RMA operations
    MPI_Win_fence(MPI_MODE_NOPRECEDE, windows[n]);
  }


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

  // Tell MPI we are done with RMA
  for (unsigned n=0; n<nd; ++n)
    MPI_Win_fence(MPI_MODE_NOPUT+MPI_MODE_NOSTORE+MPI_MODE_NOSUCCEED, windows[n]);
#endif
}

// template <typename ExecSpace>
// void
// KtensorOneSidedUpdate<ExecSpace>::
// doImportSparse(const KtensorT<ExecSpace>& u_overlapped,
//                const KtensorT<ExecSpace>& u) const
// {
// #ifdef HAVE_DIST
//   copyToWindows(u);
//   //lockWindows();

//   // Only fill u_overlapped[n] for the rows corresponding to nonzeros in X
//   const unsigned nd = u.ndims();
//   for (unsigned n=0; n<nd; ++n) {
//     const ttb_indx nrow = maps[n].capacity();
//     for (ttb_indx i=0; i<nrow; ++i) {
//       if (maps[n].valid_at(i)) {
//         const ttb_indx row = maps[n].key_at(i);
//         importRow(n, row, u, u_overlapped);
//       }
//     }
//   }

//   //unlockWindows();
//   fenceWindows();
// #endif
// }

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
  Kokkos::deep_copy(bufs[n], u[n].view());

  // Tell MPI we are going to start RMA operations
  MPI_Win_fence(MPI_MODE_NOPRECEDE, windows[n]);

  // Only fill u_overlapped[n] for the rows corresponding to nonzeros in X
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

  // Tell MPI we are done with RMA
  MPI_Win_fence(MPI_MODE_NOPUT+MPI_MODE_NOSTORE+MPI_MODE_NOSUCCEED, windows[n]);
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
    Kokkos::deep_copy(bufs[n], u[n].view());

    // Tell MPI we are going to start RMA operations
    MPI_Win_fence(MPI_MODE_NOPRECEDE, windows[n]);
  }

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

  // Tell MPI we are done with RMA
  for (unsigned n=0; n<nd; ++n)
    MPI_Win_fence(MPI_MODE_NOPUT+MPI_MODE_NOSTORE+MPI_MODE_NOSUCCEED, windows[n]);
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
  Kokkos::deep_copy(bufs[n], u[n].view());

  // Tell MPI we are going to start RMA operations
  MPI_Win_fence(MPI_MODE_NOPRECEDE, windows[n]);

  // Get data from all ranks
  for (unsigned p=0; p<np; ++p) {
    ttb_real *ptr = u_overlapped[n].view().data()+offsets_r[n][p];
    const int cnt = sizes_r[n][p];
    const MPI_Aint beg = 0;
    MPI_Get(ptr, cnt, DistContext::toMpiType<ttb_real>(), p,
            beg, cnt, DistContext::toMpiType<ttb_real>(), windows[n]);
  }

  // Tell MPI we are done with RMA
  MPI_Win_fence(MPI_MODE_NOPUT+MPI_MODE_NOSTORE+MPI_MODE_NOSUCCEED, windows[n]);
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
    Kokkos::deep_copy(bufs[n], ttb_real(0.0));

    // Tell MPI we are going to start RMA operations
    MPI_Win_fence(MPI_MODE_NOPRECEDE, windows[n]);
  }

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

  for (unsigned n=0; n<nd; ++n) {
    // Tell MPI we are done with RMA
    MPI_Win_fence(MPI_MODE_NOPUT+MPI_MODE_NOSTORE+MPI_MODE_NOSUCCEED, windows[n]);

    // Copy window data into u
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
  Kokkos::deep_copy(bufs[n], ttb_real(0.0));

  // Tell MPI we are going to start RMA operations
  MPI_Win_fence(MPI_MODE_NOPRECEDE, windows[n]);

  // Only accumulate u[n] for the rows corespondong to nonzeros in X
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

  // Tell MPI we are done with RMA
  MPI_Win_fence(MPI_MODE_NOPUT+MPI_MODE_NOSTORE+MPI_MODE_NOSUCCEED, windows[n]);

  // Copy window data into u
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
    Kokkos::deep_copy(bufs[n], ttb_real(0.0));

    // Tell MPI we are going to start RMA operations
    MPI_Win_fence(MPI_MODE_NOPRECEDE, windows[n]);
  }

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

  for (unsigned n=0; n<nd; ++n) {
    // Tell MPI we are done with RMA
    MPI_Win_fence(MPI_MODE_NOPUT+MPI_MODE_NOSTORE+MPI_MODE_NOSUCCEED, windows[n]);

    // Copy window data into u
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
  Kokkos::deep_copy(bufs[n], ttb_real(0.0));

  // Tell MPI we are going to start RMA operations
  MPI_Win_fence(MPI_MODE_NOPRECEDE, windows[n]);

  // Accumulate data to all ranks
  for (unsigned p=0; p<np; ++p) {
    const ttb_real *ptr = u_overlapped[n].view().data()+offsets_r[n][p];
    const int cnt = sizes_r[n][p];
    const MPI_Aint beg = 0;
    MPI_Accumulate(ptr, cnt, DistContext::toMpiType<ttb_real>(), p,
                   beg, cnt, DistContext::toMpiType<ttb_real>(), MPI_SUM,
                   windows[n]);
  }

  // Tell MPI we are done with RMA
  MPI_Win_fence(MPI_MODE_NOPUT+MPI_MODE_NOSTORE+MPI_MODE_NOSUCCEED, windows[n]);

  // Copy window data into u
  Kokkos::deep_copy(u[n].view(), bufs[n]);
#endif
}

#define GENTEN_PACK_DEV 1

template <typename ExecSpace>
KtensorTwoSidedUpdate<ExecSpace>::
KtensorTwoSidedUpdate(const DistTensor<ExecSpace>& X,
                      const KtensorT<ExecSpace>& u) :
  pmap(u.getProcessorMap()), nc(u.ncomponents())
{
  parallel = pmap == nullptr || (pmap != nullptr && pmap->gridSize() > 1);
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
}

template <typename ExecSpace>
KtensorTwoSidedUpdate<ExecSpace>::
~KtensorTwoSidedUpdate()
{
}

template <typename ExecSpace>
void
KtensorTwoSidedUpdate<ExecSpace>::
updateTensor(const DistTensor<ExecSpace>& X)
{
  GENTEN_TIME_MONITOR("update tensor");
  sparse = X.isSparse();

  if (sparse && parallel) {
    GENTEN_START_TIMER("initialize");
    X_sparse = X.getSptensor();
    const ttb_indx nnz = X_sparse.nnz();
    const unsigned nd = X_sparse.ndims();
    maps.resize(nd);
    num_row_sends.resize(nd);
    num_row_recvs.resize(nd);
    row_send_offsets.resize(nd);
    row_recv_offsets.resize(nd);
    row_sends.resize(nd);
    row_recvs.resize(nd);
    num_fac_sends.resize(nd);
    num_fac_recvs.resize(nd);
    fac_send_offsets.resize(nd);
    fac_recv_offsets.resize(nd);
    fac_sends.resize(nd);
    fac_recvs.resize(nd);
    row_recvs_for_proc.resize(nd);
    row_sends_dev.resize(nd);
    row_recvs_dev.resize(nd);
    fac_sends_dev.resize(nd);
    fac_recvs_dev.resize(nd);
    for (unsigned n=0; n<nd; ++n) {
      const unsigned np = offsets[n].size();
      //const ttb_indx nrows = offsets[n][np-1]+sizes[n][np-1];
      //maps[n] = unordered_map_type(nrows);
      maps[n].clear();
      //maps[n].reserve(nrows);
      num_row_sends[n].resize(np);
      num_row_recvs[n].resize(np);
      row_send_offsets[n].resize(np);
      row_recv_offsets[n].resize(np);
      num_fac_sends[n].resize(np);
      num_fac_recvs[n].resize(np);
      fac_send_offsets[n].resize(np);
      fac_recv_offsets[n].resize(np);
      row_recvs_for_proc[n].resize(np);
      for (unsigned p=0; p<np; ++p)
        row_recvs_for_proc[n][p].resize(0);

      num_row_recvs[n].clear();
      num_fac_recvs[n].clear();
    }
    GENTEN_STOP_TIMER("initialize");

    // Note:  here "sends" and "recvs" refers to the import direction of
    // communication, so "recvs" is the rows we need to receive to compute
    // the overlapped ktensor.

    // Get rows we receive from each proc
    GENTEN_START_TIMER("extract row recvs");
    auto X_sparse_h = create_mirror_view(X_sparse);
    deep_copy(X_sparse_h, X_sparse);
    for (ttb_indx i=0; i<nnz; ++i) {
      const auto subs = X_sparse_h.getSubscripts(i);
      for (unsigned n=0; n<nd; ++n) {
        const ttb_indx row = subs[n];
        //if (!maps[n].exists(row)) {
        if (maps[n].count(row) == 0) {
          unsigned p = find_proc_for_row(n, row);
          //gt_assert(!maps[n].insert(row,p).failed());
          gt_assert(maps[n].insert({row, p}).second);
          num_row_recvs[n][p] += 1;
          num_fac_recvs[n][p] += nc;
          row_recvs_for_proc[n][p].push_back(row);
        }
      }
    }
    GENTEN_STOP_TIMER("extract row recvs");

    for (unsigned n=0; n<nd; ++n) {
      const unsigned np = offsets[n].size();

      // num_recvs contains number of rows we need from each each processor,
      // get how many rows we will send to each processor
      GENTEN_START_TIMER("num row sends");
      pmap->subGridAllToAll(
        n, num_row_recvs[n].data(), 1, num_row_sends[n].data(), 1);
      GENTEN_STOP_TIMER("num row sends");

      GENTEN_START_TIMER("num fac sends");
      for (unsigned p=0; p<np; ++p)
        num_fac_sends[n][p] = num_row_sends[n][p]*nc;
      GENTEN_STOP_TIMER("num fac sends");

      // compute total number of sends and receives and offsets
      GENTEN_START_TIMER("total num sends and recvs");
      ttb_indx tot_num_row_sends = 0;
      ttb_indx tot_num_row_recvs = 0;
      ttb_indx tot_num_fac_sends = 0;
      ttb_indx tot_num_fac_recvs = 0;
      for (unsigned p=0; p<np; ++p) {
        row_send_offsets[n][p] = tot_num_row_sends;
        row_recv_offsets[n][p] = tot_num_row_recvs;
        tot_num_row_sends += num_row_sends[n][p];
        tot_num_row_recvs += num_row_recvs[n][p];

        fac_send_offsets[n][p] = tot_num_fac_sends;
        fac_recv_offsets[n][p] = tot_num_fac_recvs;
        tot_num_fac_sends += num_fac_sends[n][p];
        tot_num_fac_recvs += num_fac_recvs[n][p];
      }
      GENTEN_STOP_TIMER("total num sends and recvs");

      // allocate row send/recv buffers
      GENTEN_START_TIMER("allocate buffers");
      row_sends_dev[n] = row_vec_type("row_sends", tot_num_row_sends);
      row_recvs_dev[n] = row_vec_type("row_recvs", tot_num_row_recvs);
      fac_sends_dev[n] = fac_vec_type("fac_sends", tot_num_fac_sends);
      fac_recvs_dev[n] = fac_vec_type("fac_recvs", tot_num_fac_recvs);
      row_sends[n].resize(tot_num_row_sends);
      row_recvs[n].resize(tot_num_row_recvs);
      fac_sends[n].resize(tot_num_fac_sends);
      fac_recvs[n].resize(tot_num_fac_recvs);
      GENTEN_STOP_TIMER("allocate buffers");

      // Sort rows we receive from eaach processor for more predictable
      // memory accesses of factor matrix rows
      GENTEN_START_TIMER("sort row recvs");
      for (unsigned p=0; p<np; ++p)
        std::sort(row_recvs_for_proc[n][p].begin(),
                  row_recvs_for_proc[n][p].end());
      GENTEN_STOP_TIMER("sort row recvs");

      // fill row recv buffer with rows we receive from each proc
      GENTEN_START_TIMER("fill row recvs");
      auto rrh = Kokkos::create_mirror_view(row_recvs_dev[n]);
      ttb_indx idx = 0;
      for (unsigned p=0; p<np; ++p) {
        gt_assert(static_cast<int>(row_recvs_for_proc[n][p].size()) == num_row_recvs[n][p]);
        for (int i=0; i<num_row_recvs[n][p]; ++i) {
          row_recvs[n][idx] = row_recvs_for_proc[n][p][i];
          rrh[idx++] = row_recvs_for_proc[n][p][i];
        }
      }
      Kokkos::deep_copy(row_recvs_dev[n], rrh);
      GENTEN_STOP_TIMER("fill row recvs");

      // compute rows we need to send
      GENTEN_START_TIMER("fill row sends");
      pmap->subGridAllToAll(
        n,
        row_recvs[n].data(),num_row_recvs[n].data(),row_recv_offsets[n].data(),
        row_sends[n].data(),num_row_sends[n].data(),row_send_offsets[n].data());
      Kokkos::View<ttb_indx*,Kokkos::HostSpace> row_sends_umv(
        row_sends[n].data(), tot_num_row_sends);
      Kokkos::deep_copy(row_sends_dev[n], row_sends_umv);
      GENTEN_STOP_TIMER("fill row sends");
    }
  }
}

template <typename ExecSpace>
KtensorT<ExecSpace>
KtensorTwoSidedUpdate<ExecSpace>::
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
KtensorTwoSidedUpdate<ExecSpace>::
initOverlapKtensor(KtensorT<ExecSpace>& u) const
{
  GENTEN_TIME_MONITOR("k-tensor init");
  if (parallel && sparse) {
#if GENTEN_PACK_DEV
    const unsigned nd = u.ndims();
    const unsigned nc_ = nc;
    for (unsigned n=0; n<nd; ++n) {
      const unsigned np = pmap->subCommSize(n);
      auto rr = row_recvs_dev[n];
      auto un = u[n];
      for (unsigned p=0; p<np; ++p) {
        const unsigned nrow = num_row_recvs[n][p];
        const unsigned off = row_recv_offsets[n][p];
        Kokkos::parallel_for("Genten::TwoSidedDKU::initOverlapKtensor",
                         Kokkos::RangePolicy<ExecSpace>(0,nrow),
                         KOKKOS_LAMBDA(const unsigned i)
        {
          for (unsigned j=0; j<nc_; ++j)
            un.entry(rr[off+i],j) = 0.0;
        });
      }
    }
#else
    auto uh = create_mirror_view(u);
    const unsigned nd = u.ndims();
    for (unsigned n=0; n<nd; ++n) {
      const unsigned np = pmap->subCommSize(n);
      for (unsigned p=0; p<np; ++p) {
        const unsigned nrow = num_row_recvs[n][p];
        const unsigned off = row_recv_offsets[n][p];
        for (unsigned i=0; i<nrow; ++i)
          for (unsigned j=0; j<nc; ++j)
            uh[n].entry(row_recvs[n][off+i],j) = 0.0;
      }
    }
    deep_copy(u, uh);
#endif
  }
  else {
    u.setMatrices(0.0);
  }
  u.weights() = ttb_real(1.0);
  Kokkos::fence(); // for accurate timer
}

template <typename ExecSpace>
void
KtensorTwoSidedUpdate<ExecSpace>::
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
KtensorTwoSidedUpdate<ExecSpace>::
doImport(const KtensorT<ExecSpace>& u_overlapped,
         const KtensorT<ExecSpace>& u,
         const ttb_indx n) const
{
  GENTEN_TIME_MONITOR("k-tensor import");
  if (parallel) {
    if (sparse) {
#if GENTEN_PACK_DEV
      doImportSparseDev(u_overlapped, u, n);
#else
      doImportSparse(u_overlapped, u, n);
#endif
    }
    else
      doImportDense(u_overlapped, u, n);
  }
  else
    deep_copy(u_overlapped[n], u[n]); // no-op if u and u_overlapped are the same
}

template <typename ExecSpace>
void
KtensorTwoSidedUpdate<ExecSpace>::
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
KtensorTwoSidedUpdate<ExecSpace>::
doExport(const KtensorT<ExecSpace>& u,
         const KtensorT<ExecSpace>& u_overlapped,
         const ttb_indx n) const
{
  GENTEN_TIME_MONITOR("k-tensor export");
  if (parallel) {
    if (sparse) {
#if GENTEN_PACK_DEV
      doExportSparseDev (u, u_overlapped, n);
#else
      doExportSparse(u, u_overlapped, n);
#endif
    }
    else
      doExportDense(u, u_overlapped, n);
  }
  else
    deep_copy(u[n], u_overlapped[n]); // no-op if u and u_overlapped are the same
}

template <typename ExecSpace>
unsigned
KtensorTwoSidedUpdate<ExecSpace>::
find_proc_for_row(unsigned n, unsigned row) const
{
  // Find processor given row lives on by finding processor where
  // the first row is bigger than the given row, then move back one
  const unsigned np = pmap->subCommSize(n);
  unsigned p = 0;
  while (p < np && row >= ttb_indx(offsets[n][p])) ++p;
  --p;
  return p;
}

template <typename ExecSpace>
void
KtensorTwoSidedUpdate<ExecSpace>::
doImportSparse(const KtensorT<ExecSpace>& u_overlapped,
               const KtensorT<ExecSpace>& u) const
{
  const unsigned nd = u.ndims();
  for (unsigned n=0; n<nd; ++n) {
#if GENTEN_PACK_DEV
    doImportSparseDev(u_overlapped, u, n);
#else
    doImportSparse(u_overlapped, u, n);
#endif
  }
}

template <typename ExecSpace>
void
KtensorTwoSidedUpdate<ExecSpace>::
doImportSparse(const KtensorT<ExecSpace>& u_overlapped,
               const KtensorT<ExecSpace>& u,
               const ttb_indx n) const
{
  const unsigned rank = pmap->subCommRank(n);
  const unsigned np = pmap->subCommSize(n);

  // Check view sizes match what we expect
  gt_assert(u[n].view().span() == size_t(sizes_r[n][rank]));
  gt_assert(u_overlapped[n].view().span() == size_t(offsets_r[n][np-1]+sizes_r[n][np-1]));

  // Pack u into send buffer
  GENTEN_START_TIMER("pack-host");
  auto uhn = create_mirror_view(u[n]);
  deep_copy(uhn, u[n]);
  for (unsigned p=0; p<np; ++p) {
    const unsigned nrow = num_row_sends[n][p];
    const unsigned off = row_send_offsets[n][p];
    for (unsigned i=0; i<nrow; ++i)
      for (unsigned j=0; j<nc; ++j) {
        const ttb_indx idx = nc*(off+i)+j;
        const ttb_indx row = row_sends[n][off+i]-offsets[n][rank];
        fac_sends[n][idx] = uhn.entry(row,j);
      }
  }
  GENTEN_STOP_TIMER("pack-host");

  // Import off-processor rows
  GENTEN_START_TIMER("communication");
  pmap->subGridAllToAll(
    n,
    fac_sends[n].data(), num_fac_sends[n].data(), fac_send_offsets[n].data(),
    fac_recvs[n].data(), num_fac_recvs[n].data(), fac_recv_offsets[n].data());
  GENTEN_STOP_TIMER("communication");

  // Copy recv buffer into u_overlapped
  GENTEN_START_TIMER("unpack-host");
  auto uhn_overlapped = create_mirror_view(u_overlapped[n]);
  for (unsigned p=0; p<np; ++p) {
    const unsigned nrow = num_row_recvs[n][p];
    const unsigned off = row_recv_offsets[n][p];
    for (unsigned i=0; i<nrow; ++i)
      for (unsigned j=0; j<nc; ++j)
        uhn_overlapped.entry(row_recvs[n][off+i],j) =
          fac_recvs[n][nc*(off+i)+j];
  }
  deep_copy(u_overlapped[n], uhn_overlapped);
  GENTEN_STOP_TIMER("unpack-host");
}

template <typename ExecSpace>
void
KtensorTwoSidedUpdate<ExecSpace>::
doImportSparseDev(const KtensorT<ExecSpace>& u_overlapped,
                  const KtensorT<ExecSpace>& u,
                  const ttb_indx n) const
{
  const unsigned rank = pmap->subCommRank(n);
  const unsigned np = pmap->subCommSize(n);

  // Check view sizes match what we expect
  gt_assert(u[n].view().span() == size_t(sizes_r[n][rank]));
  gt_assert(u_overlapped[n].view().span() == size_t(offsets_r[n][np-1]+sizes_r[n][np-1]));

  // Pack u into send buffer
  GENTEN_START_TIMER("pack-device");
  auto fs = fac_sends_dev[n];
  auto rs = row_sends_dev[n];
  auto un = u[n];
  const ttb_indx gid_offset = offsets[n][rank];
  const unsigned nc_ = nc;
  for (unsigned p=0; p<np; ++p) {
    const unsigned nrow = num_row_sends[n][p];
    const unsigned off = row_send_offsets[n][p];
    Kokkos::parallel_for("Genten::TwoSidedDKU::Import_Pack",
                         Kokkos::RangePolicy<ExecSpace>(0,nrow),
                         KOKKOS_LAMBDA(const unsigned i)
    {
      const ttb_indx row = rs[off+i]-gid_offset;
      for (unsigned j=0; j<nc_; ++j)
        fs[nc_*(off+i)+j] = un.entry(row,j);
    });
  }
  Kokkos::fence();
  GENTEN_STOP_TIMER("pack-device");

  // Import off-processor rows
  GENTEN_START_TIMER("communication");
  pmap->subGridAllToAll(
    n,
    fac_sends[n].data(), num_fac_sends[n].data(), fac_send_offsets[n].data(),
    fac_recvs[n].data(), num_fac_recvs[n].data(), fac_recv_offsets[n].data());
  GENTEN_STOP_TIMER("communication");

  // Copy recv buffer into u_overlapped
  GENTEN_START_TIMER("unpack-device");
  auto fr = fac_recvs_dev[n];
  auto rr = row_recvs_dev[n];
  auto uon = u_overlapped[n];
  for (unsigned p=0; p<np; ++p) {
    const unsigned nrow = num_row_recvs[n][p];
    const unsigned off = row_recv_offsets[n][p];
    Kokkos::parallel_for("Genten::TwoSidedDKU::Import_Unpack",
                         Kokkos::RangePolicy<ExecSpace>(0,nrow),
                         KOKKOS_LAMBDA(const unsigned i)
    {
      for (unsigned j=0; j<nc_; ++j)
        uon.entry(rr[off+i],j) = fr[nc_*(off+i)+j];
    });
  }
  Kokkos::fence();
  GENTEN_STOP_TIMER("unpack-device");
}

template <typename ExecSpace>
void
KtensorTwoSidedUpdate<ExecSpace>::
doImportDense(const KtensorT<ExecSpace>& u_overlapped,
              const KtensorT<ExecSpace>& u) const
{
  const unsigned nd = u.ndims();
  for (unsigned n=0; n<nd; ++n)
    doImportDense(u_overlapped, u, n);
}

template <typename ExecSpace>
void
KtensorTwoSidedUpdate<ExecSpace>::
doImportDense(const KtensorT<ExecSpace>& u_overlapped,
              const KtensorT<ExecSpace>& u,
              const ttb_indx n) const
{
  auto uov = u_overlapped[n].view();
  const unsigned rank = pmap->subCommRank(n);
  const unsigned np = pmap->subCommSize(n);
  gt_assert(u[n].view().span() == size_t(sizes_r[n][rank]));
  gt_assert(uov.span() == size_t(offsets_r[n][np-1]+sizes_r[n][np-1]));
  Kokkos::fence();
  pmap->subGridAllGather(n, u[n].view(), uov,
                         sizes_r[n].data(), offsets_r[n].data());
}

template <typename ExecSpace>
void
KtensorTwoSidedUpdate<ExecSpace>::
doExportSparse(const KtensorT<ExecSpace>& u,
               const KtensorT<ExecSpace>& u_overlapped) const
{
  const unsigned nd = u.ndims();
  for (unsigned n=0; n<nd; ++n) {
#if GENTEN_PACK_DEV
    doExportSparseDev(u, u_overlapped, n);
#else
    doExportSparse(u, u_overlapped, n);
#endif
  }
}

template <typename ExecSpace>
void
KtensorTwoSidedUpdate<ExecSpace>::
doExportSparse(const KtensorT<ExecSpace>& u,
               const KtensorT<ExecSpace>& u_overlapped,
               const ttb_indx n) const
{
  const unsigned rank = pmap->subCommRank(n);
  const unsigned np = pmap->subCommSize(n);

  // Check view sizes match what we expect
  gt_assert(u[n].view().span() == size_t(sizes_r[n][rank]));
  gt_assert(u_overlapped[n].view().span() == size_t(offsets_r[n][np-1]+sizes_r[n][np-1]));

  // Pack u_overlapped into recv buffer
  GENTEN_START_TIMER("pack-host");
  auto uhn_overlapped = create_mirror_view(u_overlapped[n]);
  deep_copy(uhn_overlapped, u_overlapped[n]);
  for (unsigned p=0; p<np; ++p) {
    const unsigned nrow = num_row_recvs[n][p];
    const unsigned off = row_recv_offsets[n][p];
    for (unsigned i=0; i<nrow; ++i)
      for (unsigned j=0; j<nc; ++j)
        fac_recvs[n][nc*(off+i)+j] =
          uhn_overlapped.entry(row_recvs[n][off+i],j);
  }
  GENTEN_STOP_TIMER("pack-host");

  // Export off-processor rows
  GENTEN_START_TIMER("communication");
  pmap->subGridAllToAll(
    n,
    fac_recvs[n].data(), num_fac_recvs[n].data(), fac_recv_offsets[n].data(),
    fac_sends[n].data(), num_fac_sends[n].data(), fac_send_offsets[n].data());
  GENTEN_STOP_TIMER("communication");

  // Copy send buffer into u, combining rows that are sent from multiple procs
  GENTEN_START_TIMER("unpack-host");
  auto uhn = create_mirror_view(u[n]);
  uhn = 0.0;
  for (unsigned p=0; p<np; ++p) {
    const unsigned nrow = num_row_sends[n][p];
    const unsigned off = row_send_offsets[n][p];
    for (unsigned i=0; i<nrow; ++i)
      for (unsigned j=0; j<nc; ++j)
        uhn.entry(row_sends[n][off+i]-offsets[n][rank],j)
          += fac_sends[n][nc*(off+i)+j];
  }
  deep_copy(u[n], uhn);
  GENTEN_STOP_TIMER("unpack-host");
}

template <typename ExecSpace>
void
KtensorTwoSidedUpdate<ExecSpace>::
doExportSparseDev(const KtensorT<ExecSpace>& u,
                  const KtensorT<ExecSpace>& u_overlapped,
                  const ttb_indx n) const
{
  const unsigned rank = pmap->subCommRank(n);
  const unsigned np = pmap->subCommSize(n);

  // Check view sizes match what we expect
  gt_assert(u[n].view().span() == size_t(sizes_r[n][rank]));
  gt_assert(u_overlapped[n].view().span() == size_t(offsets_r[n][np-1]+sizes_r[n][np-1]));

  // Pack u_overlapped into recv buffer
  GENTEN_START_TIMER("pack-device");
  auto fr = fac_recvs_dev[n];
  auto rr = row_recvs_dev[n];
  auto uon = u_overlapped[n];
  const unsigned nc_ = nc;
  for (unsigned p=0; p<np; ++p) {
    const unsigned nrow = num_row_recvs[n][p];
    const unsigned off = row_recv_offsets[n][p];
    Kokkos::parallel_for("Genten::TwoSidedDKU::Export_Pack",
                         Kokkos::RangePolicy<ExecSpace>(0,nrow),
                         KOKKOS_LAMBDA(const unsigned i)
    {
      for (unsigned j=0; j<nc_; ++j)
        fr[nc_*(off+i)+j] = uon.entry(rr[off+i],j);
    });
  }
  Kokkos::fence();
  GENTEN_STOP_TIMER("pack-device");

  // Export off-processor rows
  GENTEN_START_TIMER("communication");
  pmap->subGridAllToAll(
    n,
    fac_recvs[n].data(), num_fac_recvs[n].data(), fac_recv_offsets[n].data(),
    fac_sends[n].data(), num_fac_sends[n].data(), fac_send_offsets[n].data());
  GENTEN_STOP_TIMER("communication");

  // Copy send buffer into u, combining rows that are sent from multiple procs
  GENTEN_START_TIMER("unpack-device");
  auto fs = fac_sends_dev[n];
  auto rs = row_sends_dev[n];
  auto un = u[n];
  un = 0.0;
  const ttb_indx gid_offset = offsets[n][rank];
  for (unsigned p=0; p<np; ++p) {
    const unsigned nrow = num_row_sends[n][p];
    const unsigned off = row_send_offsets[n][p];
    Kokkos::parallel_for("Genten::TwoSidedDKU::Import_Unpack",
                         Kokkos::RangePolicy<ExecSpace>(0,nrow),
                         KOKKOS_LAMBDA(const unsigned i)
    {
      const ttb_indx row = rs[off+i]-gid_offset;
      for (unsigned j=0; j<nc_; ++j)
        un.entry(row,j) += fs[nc_*(off+i)+j];
    });
  }
  Kokkos::fence();
  GENTEN_STOP_TIMER("unpack-device");
}

template <typename ExecSpace>
void
KtensorTwoSidedUpdate<ExecSpace>::
doExportDense(const KtensorT<ExecSpace>& u,
              const KtensorT<ExecSpace>& u_overlapped) const
{
  const unsigned nd = u.ndims();
  for (unsigned n=0; n<nd; ++n)
    doExportDense(u, u_overlapped, n);
}

template <typename ExecSpace>
void
KtensorTwoSidedUpdate<ExecSpace>::
doExportDense(const KtensorT<ExecSpace>& u,
              const KtensorT<ExecSpace>& u_overlapped,
              const ttb_indx n) const
{
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

}

#define INST_MACRO(SPACE) \
  template class Genten::KtensorOneSidedUpdate<SPACE>; \
  template class Genten::KtensorTwoSidedUpdate<SPACE>;

GENTEN_INST(INST_MACRO)
