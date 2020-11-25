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

#include "Genten_DistContext.hpp"
#include "Genten_Boost.hpp"
#include "Genten_TensorInfo.hpp"

namespace Genten {

class ProcessorMap {
public:
  ProcessorMap() = default;
  ProcessorMap(ptree const& input_tree, TensorInfo const& info);

private:
  /*
   * FieldDecls
   */
  MPI_Comm cart_comm_; 
  int grid_nprocs_;
  int grid_rank_;

  small_vector<int> sub_grid_rank_;
  small_vector<int> dimension_sizes_;
  small_vector<MPI_Comm> sub_maps_; // N-1 D sub comms

  TensorInfo tensor_info_;
  ptree pmap_tree_;
};

} // namespace Genten
