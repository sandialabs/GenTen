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
#include "Genten_Pmap.hpp"
#include "Genten_Tpetra.hpp"

namespace Genten {

template <typename ExecSpace> class SptensorT;
template <typename ExecSpace> class TensorT;

template <typename ExecSpace>
class DistTensor {
public:
  using exec_space = ExecSpace;

  DistTensor(const ttb_indx nd) : pmap(nullptr)
  {
#ifdef HAVE_TPETRA
    factorMaps.resize(nd);
    tensorMaps.resize(nd);
    importers.resize(nd);
#endif
  }

  DistTensor() = default;
  DistTensor(DistTensor&&) = default;
  DistTensor(const DistTensor&) = default;
  DistTensor& operator=(DistTensor&&) = default;
  DistTensor& operator=(const DistTensor&) = default;
  virtual ~DistTensor() = default;

  void setProcessorMap(const ProcessorMap* pmap_) { pmap = pmap_; }
  const ProcessorMap* getProcessorMap() const { return pmap; }

  virtual bool isSparse() const { return false; }
  virtual bool isDense() const { return false; }
  virtual SptensorT<ExecSpace> getSptensor() const;
  virtual TensorT<ExecSpace> getTensor() const;

#ifdef HAVE_TPETRA
  Teuchos::RCP<const tpetra_map_type<ExecSpace> >
  factorMap(const unsigned n) const {
    gt_assert(n < factorMaps.size());
    return factorMaps[n];
  }
  Teuchos::RCP<const tpetra_map_type<ExecSpace> >
  tensorMap(const unsigned n) const {
    gt_assert(n < tensorMaps.size());
    return tensorMaps[n];
  }
  Teuchos::RCP<const tpetra_import_type<ExecSpace> >
  importer(const unsigned n) const {
    gt_assert(n < importers.size());
    return importers[n];
  }
  Teuchos::RCP<const tpetra_map_type<ExecSpace> >&
  factorMap(const unsigned n) {
    gt_assert(n < factorMaps.size());
    return factorMaps[n];
  }
  Teuchos::RCP<const tpetra_map_type<ExecSpace> >&
  tensorMap(const unsigned n) {
    gt_assert(n < tensorMaps.size());
    return tensorMaps[n];
  }
  Teuchos::RCP<const tpetra_import_type<ExecSpace> >&
  importer(const unsigned n) {
    gt_assert(n < importers.size());
    return importers[n];
  }

  const std::vector< Teuchos::RCP<const tpetra_map_type<ExecSpace> > >&
  getFactorMaps() const { return factorMaps; }
  const std::vector< Teuchos::RCP<const tpetra_map_type<ExecSpace> > >&
  getTensorMaps() const { return tensorMaps; }
  const std::vector< Teuchos::RCP<const tpetra_import_type<ExecSpace> > >&
  getImporters() const { return importers; }
  std::vector< Teuchos::RCP<const tpetra_map_type<ExecSpace> > >&
  getFactorMaps() { return factorMaps; }
  std::vector< Teuchos::RCP<const tpetra_map_type<ExecSpace> > >&
  getTensorMaps() { return tensorMaps; }
  std::vector< Teuchos::RCP<const tpetra_import_type<ExecSpace> > >&
  getImporters() { return importers; }
#endif

protected:

  const ProcessorMap* pmap;

#ifdef HAVE_TPETRA
  std::vector< Teuchos::RCP< const tpetra_map_type<ExecSpace> > > factorMaps;
  std::vector< Teuchos::RCP< const tpetra_map_type<ExecSpace> > > tensorMaps;
  std::vector< Teuchos::RCP< const tpetra_import_type<ExecSpace> > > importers;
#endif

};

}

#include "Genten_Sptensor.hpp"
#include "Genten_Tensor.hpp"

namespace Genten {

template <typename ExecSpace>
SptensorT<ExecSpace>
DistTensor<ExecSpace>::getSptensor() const { return SptensorT<ExecSpace>(); }

template <typename ExecSpace>
TensorT<ExecSpace>
DistTensor<ExecSpace>::getTensor() const { return TensorT<ExecSpace>(); }

}
