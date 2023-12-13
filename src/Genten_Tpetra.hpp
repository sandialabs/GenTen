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

#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Import.hpp"
#include "Tpetra_DistObject.hpp"
#include "Tpetra_Details_makeOptimizedColMap.hpp"

namespace Genten {

template <typename ExecSpace>
using tpetra_node_type = Tpetra::KokkosCompat::KokkosDeviceWrapperNode<ExecSpace>;

using tpetra_lo_type = int;

using tpetra_go_type = long long;

template <typename ExecSpace>
using tpetra_map_type = Tpetra::Map<tpetra_lo_type, tpetra_go_type, tpetra_node_type<ExecSpace> >;

template <typename ExecSpace>
using tpetra_multivector_type = Tpetra::MultiVector<ttb_real, tpetra_lo_type, tpetra_go_type, tpetra_node_type<ExecSpace> >;

template <typename ExecSpace>
using tpetra_import_type = Tpetra::Import<tpetra_lo_type, tpetra_go_type, tpetra_node_type<ExecSpace> >;

template <typename ExecSpace>
using tpetra_dist_object_type = Tpetra::DistObject<ttb_real, tpetra_lo_type, tpetra_go_type, tpetra_node_type<ExecSpace> >;

}

#endif
