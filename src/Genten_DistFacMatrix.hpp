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
#ifdef HAVE_TPETRA

#include "Genten_Tpetra.hpp"
#include "Genten_FacMatrix.hpp"

#include "Tpetra_Details_reallocDualViewIfNeeded.hpp"

namespace Genten {

namespace Impl {

template <typename ExecSpace, typename LocalOrdinal, typename BufferDevice>
void
copyAndPermute(
  const FacMatrixT<ExecSpace>& src,
  const FacMatrixT<ExecSpace>& dst,
  const size_t numSameIDs,
  const Kokkos::DualView<const LocalOrdinal*, BufferDevice>& permuteToLIDs,
  const Kokkos::DualView<const LocalOrdinal*, BufferDevice>& permuteFromLIDs,
  const Tpetra::CombineMode combineMode)
{
  const unsigned nc = dst.nCols();

  // Copy rows that are the same in src and dst
  if (numSameIDs > 0) {
    Kokkos::RangePolicy<ExecSpace> policy(0,numSameIDs);
    if (combineMode == Tpetra::ADD_ASSIGN) {
      Kokkos::parallel_for("Genten::DistFacMatrix::copyAndPermute::copy",
                           policy, KOKKOS_LAMBDA(const ttb_indx i)
      {
        for (unsigned j=0; j<nc; ++j)
          dst.entry(i,j) += src.entry(i,j);
      });
    }
    else {
      const std::pair<size_t, size_t> rows(0, numSameIDs);
      deep_copy(Kokkos::subview(dst.view(), rows, Kokkos::ALL),
                Kokkos::subview(src.view(), rows, Kokkos::ALL));
    }
  }

  // Permute remaining rows
  if (permuteToLIDs.extent(0) > 0) {
    Kokkos::RangePolicy<ExecSpace> policy(0,permuteToLIDs.extent(0));
    auto permTo = permuteToLIDs.view_device();
    auto permFrom = permuteFromLIDs.view_device();
    if (combineMode == Tpetra::ADD_ASSIGN) {
      Kokkos::parallel_for("Genten::DistFacMatrix::copyAndPermute::permute",
                           policy, KOKKOS_LAMBDA(const ttb_indx i)
      {
        auto row_to = permTo(i);
        auto row_from = permFrom(i);
        for (unsigned j=0; j<nc; ++j)
          dst.entry(row_to,j) += src.entry(row_from,j);
      });
    }
    else {
      Kokkos::parallel_for("Genten::DistFacMatrix::copyAndPermute::permute",
                           policy, KOKKOS_LAMBDA(const ttb_indx i)
      {
        auto row_to = permTo(i);
        auto row_from = permFrom(i);
        for (unsigned j=0; j<nc; ++j)
          dst.entry(row_to,j) = src.entry(row_from,j);
      });
    }
  }
}

template <typename ExecSpace, typename LocalOrdinal, typename BufferDevice>
void
packAndPrepare(
  const FacMatrixT<ExecSpace>& src,
  const FacMatrixT<ExecSpace>& dst,
  const Kokkos::DualView<const LocalOrdinal*, BufferDevice>& exportLIDs,
  Kokkos::DualView<ttb_real*, BufferDevice>& exports,
  Kokkos::DualView<size_t*, BufferDevice> numPacketsPerLID,
  size_t& constantNumPackets)
{
  const unsigned nc = dst.nCols();
  constantNumPackets = nc;

  const size_t numExports = exportLIDs.extent(0);
  if (numExports > 0) {
    const size_t exportsSize = numExports*nc;
    Tpetra::Details::reallocDualViewIfNeeded(exports, exportsSize, "exports");
    Kokkos::RangePolicy<ExecSpace> policy(0, numExports);
    exports.clear_sync_state();
    exports.modify_device();
    auto e = exports.view_device();
    auto e_lids = exportLIDs.view_device();
    Kokkos::parallel_for("Genten::DistFacMatrix::packAndPrepare",
                         policy, KOKKOS_LAMBDA(const ttb_indx i)
    {
      auto row = e_lids(i);
      for (unsigned j=0; j<nc; ++j)
        e(i*nc+j) = src.entry(row,j);
    });
  }
}

template <typename ExecSpace, typename LocalOrdinal, typename BufferDevice>
void
unpackAndCombine(
  const FacMatrixT<ExecSpace>& dst,
  const Kokkos::DualView<const LocalOrdinal*, BufferDevice>& importLIDs,
  Kokkos::DualView<ttb_real*, BufferDevice>& imports,
  Kokkos::DualView<size_t*, BufferDevice> numPacketsPerLID,
  const size_t constantNumPackets,
  const Tpetra::CombineMode combineMode)
{
  using space_prop = SpaceProperties<ExecSpace>;
  const unsigned nc = dst.nCols();
  const size_t numImports = importLIDs.extent(0);
  if (numImports > 0) {
    Kokkos::RangePolicy<ExecSpace> policy(0, numImports);
    imports.sync_device();
    auto im = imports.view_device();
    auto im_lids = importLIDs.view_device();
    if (combineMode == Tpetra::ADD || combineMode == Tpetra::ADD_ASSIGN) {
      if (space_prop::concurrency() > 1) {
        Kokkos::parallel_for(
          "Genten::DistFacMatrix::unpackAndCombine::add_atomic",
          policy, KOKKOS_LAMBDA(const ttb_indx i)
        {
          auto row = im_lids(i);
          for (unsigned j=0; j<nc; ++j)
            Kokkos::atomic_add(&(dst.entry(row,j)), im(i*nc+j));
        });
      }
      else {
        Kokkos::parallel_for(
          "Genten::DistFacMatrix::unpackAndCombine::add",
          policy, KOKKOS_LAMBDA(const ttb_indx i)
        {
          auto row = im_lids(i);
          for (unsigned j=0; j<nc; ++j)
            dst.entry(row,j) += im(i*nc+j);
        });
      }
    }
    else if (combineMode == Tpetra::INSERT || combineMode == Tpetra::REPLACE) {
      Kokkos::parallel_for(
        "Genten::DistFacMatrix::unpackAndCombine::insert",
        policy, KOKKOS_LAMBDA(const ttb_indx i)
      {
        auto row = im_lids(i);
        for (unsigned j=0; j<nc; ++j)
          dst.entry(row,j) = im(i*nc+j);
      });
    }
    else
      Genten::error("DistFacMatrix only supports ADD and INSERT combine modes");
  }
}

}

template <typename ExecSpace>
class DistFacMatrix : public tpetra_dist_object_type<ExecSpace> {
public:
  using dist_object_type = tpetra_dist_object_type<ExecSpace>;
  using local_ordinal_type = typename dist_object_type::local_ordinal_type;
  using global_ordinal_type = typename dist_object_type::global_ordinal_type;
  using node_type = typename dist_object_type::node_type;
  using device_type = typename dist_object_type::device_type;
  using execution_space = typename dist_object_type::execution_space;
  using map_type = typename dist_object_type::map_type;

  DistFacMatrix(const FacMatrixT<ExecSpace>& mat_,
                const Teuchos::RCP< const tpetra_map_type<ExecSpace> >& map) :
    dist_object_type(map), mat(mat_) {}

  virtual ~DistFacMatrix() {}

protected:
  using buffer_device_type = typename dist_object_type::buffer_device_type;
  FacMatrixT<ExecSpace> mat;

  virtual bool
  checkSizes(const Tpetra::SrcDistObject& src) override
  {
    const DistFacMatrix* src_dist_fac_mat =
      dynamic_cast<const DistFacMatrix*>(&src);
    if (src_dist_fac_mat == nullptr)
      return false;
    return src_dist_fac_mat->mat.nCols() == mat.nCols();
  }

  // Bring in DistObject overloaded implementations to prevent warnings about
  // partially overloaded functions
  using dist_object_type::copyAndPermute;
  using dist_object_type::packAndPrepare;
  using dist_object_type::unpackAndCombine;

  virtual void
  copyAndPermute(
    const Tpetra::SrcDistObject& src,
    const size_t numSameIDs,
    const Kokkos::DualView<const local_ordinal_type*, buffer_device_type>& permuteToLIDs,
    const Kokkos::DualView<const local_ordinal_type*, buffer_device_type>& permuteFromLIDs,
    const Tpetra::CombineMode CM) override
  {
    //GENTEN_TIME_MONITOR("copyAndPermute");

    const DistFacMatrix& src_dist_fac_mat =
      dynamic_cast<const DistFacMatrix&>(src);
    Impl::copyAndPermute(src_dist_fac_mat.mat, mat, numSameIDs, permuteToLIDs,
                         permuteFromLIDs, CM);
  }

  virtual void
  packAndPrepare(
    const Tpetra::SrcDistObject& src,
    const Kokkos::DualView<const local_ordinal_type*, buffer_device_type>& exportLIDs,
    Kokkos::DualView<ttb_real*, buffer_device_type>& exports,
    Kokkos::DualView<size_t*, buffer_device_type> numPacketsPerLID,
    size_t& constantNumPackets) override
  {
    //GENTEN_TIME_MONITOR("packAndPrepare");

    const DistFacMatrix& src_dist_fac_mat =
      dynamic_cast<const DistFacMatrix&>(src);
    Impl::packAndPrepare(src_dist_fac_mat.mat, mat, exportLIDs, exports,
                         numPacketsPerLID, constantNumPackets);
  }

  virtual void
  unpackAndCombine(
    const Kokkos::DualView<const local_ordinal_type*, buffer_device_type>& importLIDs,
    Kokkos::DualView<ttb_real*, buffer_device_type> imports,
    Kokkos::DualView<size_t*, buffer_device_type> numPacketsPerLID,
    const size_t constantNumPackets,
    const Tpetra::CombineMode combineMode) override
  {
    //GENTEN_TIME_MONITOR("unpackAndCombine");

    Impl::unpackAndCombine(mat, importLIDs, imports, numPacketsPerLID,
                           constantNumPackets, combineMode);
  }

};

}

#endif
