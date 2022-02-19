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
#include "Genten_MPI_IO.h"
#include "Genten_Pmap.hpp"
#include "Genten_SpTn_Util.h"
#include "Genten_Sptensor.hpp"

#include <cmath>
#include <fstream>
#include <memory>
#include <random>
#include <unordered_map>

namespace Genten {

namespace detail {
void printGrids(ProcessorMap const &pmap);
void printBlocking(ProcessorMap const &pmap,
                   std::vector<small_vector<int>> const &blocking);

std::vector<small_vector<int>>
generateUniformBlocking(std::vector<std::uint32_t> const &ModeLengths,
                        small_vector<int> const &ProcGridSizes);
} // namespace detail

struct RangePair {
  int64_t lower;
  int64_t upper;
};

// Class to hold a block of the tensor and the factor matrix blocks that can
// be used to generate a representation of the tensor block, if the entire
// tensor is placed into one block this is just a TensorSystem
template <typename ElementType, typename ExecSpace = DefaultExecutionSpace>
class DistSpTensor {
  static_assert(std::is_floating_point<ElementType>::value,
                "DistSpTensor Requires that the element type be a floating "
                "point type.");

  /// The normal initialization method, inializes in parallel
  void init_distributed(std::string const &file_name, int indexbase);

public:
  DistSpTensor(ptree const &tree);
  ~DistSpTensor() = default;

  DistSpTensor(DistSpTensor &&) = default;
  DistSpTensor &operator=(DistSpTensor &&) = default;
  DistSpTensor(DistSpTensor const &) = default;
  DistSpTensor &operator=(DistSpTensor const &) = default;

  ElementType getTensorNorm() const;

  std::uint64_t globalNNZ() const { return Ti_.nnz; }
  std::int64_t localNNZ() const { return sp_tensor_.nnz(); }
  ElementType localNumel() const { return sp_tensor_.numel_float(); }
  std::int32_t ndims() const { return sp_tensor_.ndims(); }
  std::vector<std::uint32_t> const &dims() const { return Ti_.dim_sizes; }
  std::int64_t nprocs() const { return pmap_ptr_->gridSize(); }
  std::int64_t gridRank() const { return pmap_ptr_->gridRank(); }

  SptensorT<ExecSpace>& localSpTensor() { return sp_tensor_; }
  SptensorT<ExecSpace> const &localSpTensor() const { return sp_tensor_; }
  ProcessorMap const &pmap() const { return *pmap_ptr_; }
  std::shared_ptr<const ProcessorMap> pmap_ptr() const { return pmap_ptr_; }

  TensorInfo const &getTensorInfo() const { return Ti_; }

  std::vector<small_vector<int>> getBlocking() const { return global_blocking_; }

private:
  boost::optional<std::pair<MPI_IO::SptnFileHeader, MPI_File>>
  readHeader(std::string const &file_name, int indexbase);

  MPI_Datatype mpiElemType_ = DistContext::toMpiType<ElementType>();
  ptree input_;
  small_vector<RangePair> range_;
  SptensorT<ExecSpace> sp_tensor_;
  std::shared_ptr<ProcessorMap> pmap_ptr_;
  TensorInfo Ti_;
  bool dump_; // I don't love keeping this flag, but it's easy
  std::vector<small_vector<int>> global_blocking_;
};

// Helper declerations
namespace detail {
bool fileFormatIsBinary(std::string const &file_name);

template <typename ExecSpace>
auto rangesToIndexArray(small_vector<RangePair> const &ranges);
small_vector<int> singleDimUniformBlocking(int ModeLength, int ProcsInMode);

std::vector<MPI_IO::TDatatype<double>>
distributeTensorToVectors(std::ifstream &ifs, uint64_t nnz, int indexbase,
                          MPI_Comm comm, int rank, int nprocs);

std::vector<MPI_IO::TDatatype<double>>
redistributeTensor(std::vector<MPI_IO::TDatatype<double>> const &Tvec,
                   std::vector<std::uint32_t> const &TensorDims,
                   std::vector<small_vector<int>> const &blocking,
                   ProcessorMap const &pmap);

template <typename ExecSpace>
void printRandomElements(SptensorT<ExecSpace> const &tensor,
                         int num_elements_per_rank, ProcessorMap const &pmap,
                         small_vector<RangePair> const &ranges);
} // namespace detail

template <typename ElementType, typename ExecSpace>
DistSpTensor<ElementType, ExecSpace>::DistSpTensor(ptree const &tree)
    : input_(tree.get_child("tensor")), dump_(tree.get<bool>("dump", false)) {

  if (dump_) {
    if (DistContext::rank() == 0) {
      std::cout << "tensor:\n"
                   "\tfile: The input file\n"
                   "\tindexbase: Value that indices start at (defaults to 0)\n";
    }
    return;
  }
  const auto file_name = input_.get<std::string>("file");
  const auto indexbase = input_.get<int>("indexbase", 0);
  init_distributed(file_name, indexbase);
}

// We'll put this down here to save some space
template <typename ElementType, typename ExecSpace>
void DistSpTensor<ElementType, ExecSpace>::init_distributed(
    std::string const &file_name, int indexbase) {

  auto binary_header = readHeader(file_name, indexbase);

  const auto ndims = Ti_.dim_sizes.size();
  pmap_ptr_ = std::shared_ptr<ProcessorMap>(
      new ProcessorMap(DistContext::input(), Ti_.dim_sizes));
  auto &pmap_ = *pmap_ptr_;

  detail::printGrids(pmap_);

  const auto blocking =
      detail::generateUniformBlocking(Ti_.dim_sizes, pmap_.gridDims());

  detail::printBlocking(pmap_, blocking);
  DistContext::Barrier();

  auto t2 = MPI_Wtime();
  auto Tvec = [&] {
    if (binary_header) {
      return MPI_IO::parallelReadElements(DistContext::commWorld(),
                                          binary_header->second,
                                          binary_header->first);
    } else {
      auto tensor_file = std::ifstream(file_name);
      return detail::distributeTensorToVectors(
          tensor_file, Ti_.nnz, indexbase, pmap_.gridComm(), pmap_.gridRank(),
          pmap_.gridSize());
    }
  }();
  DistContext::Barrier();
  auto t3 = MPI_Wtime();

  if (gridRank() == 0) {
    std::cout << "Read in file in: " << t3 - t2 << "s" << std::endl;
  }

  DistContext::Barrier();
  auto t4 = MPI_Wtime();

  // Now redistribute to final format
  auto distributedData =
      detail::redistributeTensor(Tvec, Ti_.dim_sizes, blocking, pmap_);

  DistContext::Barrier();
  auto t5 = MPI_Wtime();

  if (gridRank() == 0) {
    std::cout << "Redistributied file in: " << t5 - t4 << "s" << std::endl;
  }

  DistContext::Barrier();
  auto t6 = MPI_Wtime();

  for (auto i = 0; i < ndims; ++i) {
    auto coord = pmap_.gridCoord(i);
    range_.push_back({blocking[i][coord], blocking[i][coord + 1]});
  }

  std::vector<ttb_indx> indices(ndims);
  for (auto i = 0; i < ndims; ++i) {
    auto const &rpair = range_[i];
    indices[i] = rpair.upper - rpair.lower;
  }

  const auto local_nnz = distributedData.size();
  std::vector<ttb_real> values(local_nnz);
  std::vector<std::vector<ttb_indx>> subs(local_nnz);
  for (auto i = 0; i < local_nnz; ++i) {
    auto data = distributedData[i];
    values[i] = data.val;
    subs[i] = std::vector<ttb_indx>(data.coo, data.coo + ndims);
    for (auto j = 0; j < ndims; ++j) {
      subs[i][j] -= range_[j].lower;
    }
  }

  sp_tensor_ = SptensorT<ExecSpace>(indices, values, subs);
  if (DistContext::isDebug()) {
    if (gridRank() == 0) {
      std::cout << "MPI Ranks in each dimension: ";
      for (auto p : pmap_.subCommSizes()) {
        std::cout << p << " ";
      }
      std::cout << std::endl;
    }
    DistContext::Barrier();
  }

  DistContext::Barrier();
  auto t7 = MPI_Wtime();

  if (gridRank() == 0) {
    std::cout << "Copied to data struct in: " << t7 - t6 << "s" << std::endl;
  }

  global_blocking_ = blocking;
}

template <typename ElementType, typename ExecSpace>
ElementType DistSpTensor<ElementType, ExecSpace>::getTensorNorm() const {
  auto const &values = sp_tensor_.getValArray();
  ElementType norm2 = values.dot(values);
  MPI_Allreduce(MPI_IN_PLACE, &norm2, 1, DistContext::toMpiType<ElementType>(),
                MPI_SUM, pmap_ptr_->gridComm());
  return std::sqrt(ElementType(norm2));
}

template <typename ElementType, typename ExecSpace>
boost::optional<std::pair<MPI_IO::SptnFileHeader, MPI_File>>
DistSpTensor<ElementType, ExecSpace>::readHeader(std::string const &file_name,
                                                 int indexbase) {

  bool is_binary = detail::fileFormatIsBinary(file_name);
  if (is_binary && indexbase != 0) {
    throw std::logic_error(
        "The binary format only supports zero based indexing\n");
  }

  if (!is_binary) {
    std::ifstream tensor_file(file_name);
    Ti_ = read_sptensor_header(tensor_file);
    return boost::none;
  }

  auto *mpi_fh = MPI_IO::openFile(DistContext::commWorld(), file_name);
  auto binary_header = MPI_IO::readHeader(DistContext::commWorld(), mpi_fh);
  Ti_ = binary_header.toTensorInfo();
  return std::make_pair(std::move(binary_header), mpi_fh);
}

namespace detail {

template <typename ExecSpace>
void printRandomElements(SptensorT<ExecSpace> const &tensor,
                         int num_elements_per_rank, ProcessorMap const &pmap,
                         small_vector<RangePair> const &ranges) {
  static_assert(
      std::is_same<ExecSpace, Kokkos::DefaultHostExecutionSpace>::value,
      "To print random elements we want a host tensor.");

  const auto size = pmap.gridSize();
  const auto rank = pmap.gridRank();
  auto *gComm = pmap.gridComm();

  const auto nnz = tensor.nnz();
  std::uniform_int_distribution<> dist(0, nnz - 1);
  std::mt19937_64 gen(std::random_device{}());

  for (auto i = 0; i < size; ++i) {
    if (rank != i) {
      continue;
    }
    std::cout << "Rank: " << rank << " ranges:[";
    for (auto j = 0; j < ranges.size(); ++j) {
      std::cout << "{" << ranges[j].lower << ", " << ranges[j].upper << "}";
      if (j < ranges.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]\n";
    if (nnz >= num_elements_per_rank) {
      for (auto i = 0; i < num_elements_per_rank; ++i) {
        auto rand_idx = dist(gen);
        auto indices = tensor.getSubscripts(rand_idx);
        auto value = tensor.value(rand_idx);

        std::cout << "\t";
        for (auto j = 0; j < tensor.ndims(); ++j) {
          std::cout << indices[j] + ranges[j].lower << " ";
        }
        std::cout << value << "\n";
      }
    } else {
      std::cout << "Rank: " << pmap.gridRank() << " had 0 nnz\n";
    }
    std::cout << std::endl;
    MPI_Barrier(gComm);
    sleep(1);
  }
}

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

} // namespace detail

} // namespace Genten
