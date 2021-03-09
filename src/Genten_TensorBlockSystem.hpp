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
#include "Genten_IOtext.hpp"
#include "Genten_Pmap.hpp"
#include "Genten_Sptensor.hpp"

#include <fstream>
#include <memory>

namespace Genten {

struct RangePair {
  int64_t lower;
  int64_t upper;
};

template <typename ExecSpace>
auto rangesToIndexArray(small_vector<RangePair> const &ranges) {
  IndxArrayT<ExecSpace> outArray(ranges.size());
  auto mirrorArray = create_mirror_view(outArray);

  auto i = 0;
  for (auto const &rp : ranges) {
    const auto size = rp.upper - rp.lower;
    mirrorArray[i] = size;
    ++i;
  }

  deep_copy(outArray, mirrorArray);
  return outArray;
}

// Class to hold a block of the tensor and the factor matrix blocks that can
// be used to generate a representation of the tensor block, if the entire
// tensor is placed into one block this is just a TensorSystem
template <typename ElementType, typename ExecSpace = DefaultExecutionSpace>
class TensorBlockSystem {
  static_assert(std::is_floating_point<ElementType>::value,
                "DistSpSystem Requires that the element type be a floating "
                "point type.");

  bool fileFormatIsBinary(std::string const &file_name) { 
    std::ifstream tensor_file(file_name, std::ios::binary);
    std::string header;
    header.resize(4);
    try{
      tensor_file.read(&header[0], 4);
    } catch(...){
      return false;
    }
     
    if(header == "sptn"){
      return true;
    } 

    return false; 
  }

  void init_distributed(std::string const &file_name, int indexbase, int rank);

  // Initializaton for creating the tensor block system all on all ranks
  // this will hammer the file in a bad way if it's called on every rank
  void init_independent(std::string const &file_name, int indexbase, int rank) {
    if (fileFormatIsBinary(file_name)) {
      throw std::logic_error(
          "I can't quite read the binary format just yet, sorry.");
    }

    typename SptensorT<ExecSpace>::HostMirror sp_tensor_host;
    import_sptensor(file_name, sp_tensor_host, indexbase, false, false);
    std::cout << "The size of the sp_tensor is: " << sp_tensor_host.size()
              << "\n";
    sp_tensor_ = create_mirror_view(ExecSpace{}, sp_tensor_host);
    deep_copy(sp_tensor_, sp_tensor_host);

    auto const &index_view = sp_tensor_host.size();
    const auto ndims = index_view.size();
    range_.reserve(ndims);
    for (auto i = 0; i < ndims; ++i) {
      range_.push_back({0ll, int64_t(index_view[i])});
    }

    Kfac_ =
        KtensorT<ExecSpace>(rank, ndims, rangesToIndexArray<ExecSpace>(range_));
    Kfac_.setMatrices(0.0);
  }

public:
  // This constructor is for testing and experimentation in the case that the
  // entire tensor is to be fit into a single block.  Doing that is a really
  // useful sanity check.
  TensorBlockSystem(ptree const &tree) {
    const auto file_name = tree.get<std::string>("tensor.file");
    const auto indexbase = tree.get<int>("tensor.indexbase", 0);
    const auto rank = tree.get<int>("tensor.rank", 5);

    const auto init_strategy =
        tree.get<std::string>("tensor.initialization", "distributed");
    if (init_strategy == "distributed") {
      init_distributed(file_name, indexbase, rank);
    } else if (init_strategy == "replicated") {
      // TODO This should really do single then bcast
      init_independent(file_name, indexbase, rank);
    } else if (init_strategy == "single") {
      if (DistContext::rank() == 0) {
        init_independent(file_name, indexbase, rank);
      }
    } else {
      throw std::logic_error("Tensor initialization must be one of of "
                             "{distributed, replicated, or single}\n");
    }
  }

  // Factor matrices in the ith dimension
  FacMatrixT<ExecSpace> const &factor(int i) const { return Kfac_[i]; }
  FacMatrixT<ExecSpace> &factor(int i) { return Kfac_[i]; }

private:
  small_vector<RangePair> range_;
  SptensorT<ExecSpace> sp_tensor_;
  KtensorT<ExecSpace> Kfac_;
  std::unique_ptr<ProcessorMap> pmap_ptr_;
};

namespace detail {
small_vector<int> singleDimMediumGrainBlocking(int ModeLength, int ProcsInMode);

std::vector<small_vector<int>>
generateMediumGrainBlocking(std::vector<int> ModeLengths,
                            small_vector<int> const &ProcGridSizes);

struct TDatatype {
  int coo[12] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
  double val;
};

std::vector<TDatatype> distributeTensorToVectors(std::ifstream &ifs,
                                                 uint64_t nnz, int indexbase,
                                                 MPI_Comm comm, int rank,
                                                 int nprocs);

std::vector<TDatatype> redistributeTensor(std::vector<TDatatype> const &Tvec,
                                          std::vector<int> const &TensorDims,
                                          ProcessorMap const &pmap);

} // namespace detail

// We'll put this down here to save some space
template <typename ElementType, typename ExecSpace>
void TensorBlockSystem<ElementType, ExecSpace>::init_distributed(
    std::string const &file_name, int indexbase, int rank) {
  if (fileFormatIsBinary(file_name)) {
    throw std::logic_error(
        "I can't quite read the binary format just yet, sorry.");
  }

  // TODO Bcast Ti so we don't read it on every node
  std::ifstream tensor_file(file_name);
  TensorInfo Ti = read_sptensor_header(tensor_file);
  pmap_ptr_ =
      std::unique_ptr<ProcessorMap>(new ProcessorMap(DistContext::input(), Ti));
  auto &pmap_ = *pmap_ptr_;
  MPI_Barrier(DistContext::commWorld());

  // Evenly distribute the tensor around the world
  auto Tvec = detail::distributeTensorToVectors(
      tensor_file, Ti.nnz, indexbase, pmap_.gridComm(), pmap_.gridRank(),
      pmap_.gridSize());

  // Now redistribute to medium grain format
  auto distributedData = detail::redistributeTensor(Tvec, Ti.dim_sizes, pmap_);
}

} // namespace Genten
