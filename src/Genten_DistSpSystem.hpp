//@HEADER
// ************************************************************************
//     Genten: Software for Generalized Tensor Decompositions
//     by Sandia National Laboratories
//
// Sandia National Laboratories is a multimission laboratory managed
// and operated by National Technology and Engineering Solutions of Sandia,
// LLC, a wholly owned subsidiary of Honeywell International, Inc., for the
// U.S. Department of Energy's National Nuclear Security Administration under
// contract DE-NA0003525.
//
// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
// Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// ************************************************************************
//@HEADER

#include "Genten_DistContext.hpp"
#include "Genten_Pmap.hpp"
#include "Genten_TensorInfo.hpp"

#include <iostream>

namespace Genten {
template <typename ElementType, typename ExecSpace> class DistSpSystem {

  static_assert(std::is_floating_point<ElementType>::value,
                "DistSpSystem Requires that the element type be a floating "
                "point type.");

public:
  /*
   * Constructors
   */
  DistSpSystem() = default;
  DistSpSystem(ptree const &tree) {
    tensor_tree_ = tree.get_child("tensor");
    tensor_tree_.put("order", 4);
    info_ = {100, {1, 3, 6, 8}};
    ProcessorMap pmap(tree, info_);
  }

private:
  /*
   * FieldDecls
   */
  TensorInfo info_;
  int32_t system_rank_ = 5;            // Default to a rank of 5
  boost::optional<ElementType> score_; // Or loss, we can change the name
  ptree tensor_tree_;

  /*
   * Private methods
   */
  int32_t get_system_rank() {
    if (auto system_rank = tensor_tree_.get_optional<int32_t>("tensor.rank")) {
      return system_rank.value();
    } else {
      std::cout
          << "DistSpSystem: Tensor rank was not provided, default value is: "
          << system_rank_ << ".\n";
      return system_rank_;
    }
  }
};

} // namespace Genten
