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
#pragma once

#include "Genten_Boost.hpp"
#include "Genten_DistContext.hpp"
#include "Genten_Pmap.hpp"
#include "Genten_TensorInfo.hpp"
#include "Genten_Util.hpp"
#include "Genten_DistIO.hpp"

#include <vector>

namespace Genten {
namespace detail {

// TODO We probably want to make a DistSpSystem base class that provides these
// methods independent of the element type and ExecSpace that way they aren't
// just chilling in this detail namespace passing the same things back over and
// over. But for the POC stage let's just do it this way.

// Reads just the header of the tensor to get info for blocking
TensorInfo
readTensorHeader(boost::optional<std::string> const &tensor_file_name);

void tensorPlayGround(boost::optional<std::string> const &tensor_file_name,
                      TensorInfo const &Ti, ProcessorMap const &pmap,
                      std::vector<small_vector<int>> const &blocking);

enum class TensorBlockingStrategy {
  // One block per processor in each dimension
  Uniform_OnePerRank_BS
};

// Convert a named string into a enum decleration, I kind of want to inline
// this, but I also want to keep the header easy to read ... I suspect this has
// 0 performance implications so let's move it to cpp for now.
TensorBlockingStrategy readBlockingStrategy(std::string name);

std::vector<small_vector<int>>
generateBlocking(TensorInfo const &Ti, small_vector<int> const &PmapGrid,
                 TensorBlockingStrategy Bs);

// For now assume medium grained decomposotion, this means that we can figure
// out the processor owner based solely on the range information As soon as we
// allow other distributions like multiple blocks per rank this gets more
// complicated, but for POC let's just do this.
int rankInGridThatOwns(small_vector<int> const &COO, ProcessorMap const &pmap,
                       std::vector<small_vector<int>> const &ElementRanges);
} // namespace detail

template <typename ElementType, typename ExecSpace = DefaultExecutionSpace>
class DistSpSystem {

  static_assert(std::is_floating_point<ElementType>::value,
                "DistSpSystem Requires that the element type be a floating "
                "point type.");

public:

  /*
   * Constructors
   */
  DistSpSystem() = default;
  DistSpSystem(ptree const &tree)
      : tensor_info_(detail::readTensorHeader(
            tree.get_optional<std::string>("tensor.file"))),
        pmap_(tree, tensor_info_),
        system_rank_(tree.get<int>("tensor.rank", 5)),
        tensor_tree_(tree.get_child("tensor", ptree{})) {

    auto blockingStrat = tensor_tree_.get_optional<std::string>("blocking");
    if (!blockingStrat) {
      if (DistContext::rank() == 0) {
        std::cout << "No tensor blocking strategy provided using default.\n";
      }
    }

    ranges_ = detail::generateBlocking(
        tensor_info_, pmap_.subGridSizes(),
        detail::readBlockingStrategy(blockingStrat.value_or("default")));

    detail::tensorPlayGround(tensor_tree_.get_optional<std::string>("file"),
                             tensor_info_, pmap_, ranges_);
  }

  int64_t nnz() const { return tensor_info_.nnz; }
  int tensorModeSize(int d) const { return tensor_info_.dim_sizes[d]; }
  std::vector<int> const &tensorModeSizes() const {
    return tensor_info_.dim_sizes;
  }

private:
  /*
   * FieldDecls
   */
  TensorInfo tensor_info_;
  ProcessorMap pmap_;
  int system_rank_;                    
  boost::optional<ElementType> score_; // Or loss, we can change the name
  ptree tensor_tree_;

  // Holds the blocking information for the tensor and factors
  std::vector<small_vector<int>> ranges_;

};

} // namespace Genten
