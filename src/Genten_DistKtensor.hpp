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

#include "Genten_Ktensor.hpp"
#include "Genten_Pmap.hpp"
#include "Genten_DistContext.hpp"
#include "Genten_DistSpTensor.hpp"
#include "Genten_IOtext.hpp"

namespace Genten {

template <typename ElementType, typename ExecSpace = DefaultExecutionSpace>
class DistKtensor {
  static_assert(std::is_floating_point<ElementType>::value,
                "DistKtensor Requires that the element type be a floating "
                "point type.");
public:
  DistKtensor(const DistSpTensor<ElementType,ExecSpace>& sp_tensor,
              const ptree& tree);
  ~DistKtensor() = default;

  DistKtensor(const DistKtensor&) = default;
  DistKtensor(DistKtensor&&) = default;
  DistKtensor& operator=(const DistKtensor&) = default;
  DistKtensor& operator=(DistKtensor&&) = default;

  KtensorT<ExecSpace>& localKtensor() { return ktensor_; }
  KtensorT<ExecSpace> const &localKtensor() const { return ktensor_; }
  std::int32_t rank() const { return ktensor_.ncomponents(); }
  std::int32_t ndims() const { return ktensor_.ndims(); }
  std::vector<std::uint32_t> const &global_dims() const { return ti_.dim_sizes; }

  void exportFromRoot(const KtensorT<ExecSpace>& u) const;
  KtensorT<ExecSpace> importToRoot() const;
  void allReduce(const bool divide_by_grid_size = true);
  void allReduce(KtensorT<ExecSpace>& u,
                 const bool divide_by_grid_size = true) const;
  void export_ktensor(const std::string& file_name) const;

private:
  void init_factors(const DistSpTensor<ElementType,ExecSpace>& sp_tensor);

  ptree input_;
  KtensorT<ExecSpace> ktensor_;
  std::shared_ptr<const ProcessorMap> pmap_;
  TensorInfo ti_;
  std::vector<small_vector<int>> blocking_;
};

template <typename ElementType, typename ExecSpace>
DistKtensor<ElementType, ExecSpace>::
DistKtensor(const DistSpTensor<ElementType,ExecSpace>& sp_tensor,
            const ptree& tree) :
  input_(tree.get_child("k-tensor"))
{
  ttb_indx rank = input_.get<int>("rank");
  ttb_indx nd = sp_tensor.ndims();
  ktensor_ = KtensorT<ExecSpace>(rank, nd, sp_tensor.localSpTensor().size());
  pmap_ = sp_tensor.pmap_ptr();
  ti_ = sp_tensor.getTensorInfo();
  blocking_ = sp_tensor.getBlocking();

  init_factors(sp_tensor);
}

template <typename ElementType, typename ExecSpace>
void
DistKtensor<ElementType, ExecSpace>::
exportFromRoot(const KtensorT<ExecSpace>& u) const
{
  // Broadcast ktensor values from 0 to all procs
  const ttb_indx nd = u.ndims();
  for (int i=0; i<nd; ++i)
    pmap_->gridBcast(u[i].view().data(), u[i].view().span(), 0);
  pmap_->gridBcast(u.weights().values().data(), u.weights().values().span(), 0);
  pmap_->gridBarrier();

  // Copy our portion from u into ktensor_
  ktensor_.setMatrices(0.0);
  deep_copy(ktensor_.weights(), u.weights());
  for (int i=0; i<nd; ++i) {
    auto coord = pmap_->gridCoord(i);
    auto rng = std::make_pair(blocking_[i][coord], blocking_[i][coord + 1]);
    auto sub = Kokkos::subview(u[i].view(), rng, Kokkos::ALL);
    deep_copy(ktensor_[i].view(), sub);
  }
}

template <typename ElementType, typename ExecSpace>
KtensorT<ExecSpace>
DistKtensor<ElementType, ExecSpace>::
importToRoot() const
{
  const bool print =
    DistContext::isDebug() && (pmap_->gridRank() == 0);

  KtensorT<ExecSpace> out;
  auto const &sizes = ti_.dim_sizes;
  IndxArrayT<ExecSpace> sizes_idx(sizes.size());
  for (auto i = 0; i < sizes.size(); ++i) {
    sizes_idx[i] = sizes[i];
  }
  out = KtensorT<ExecSpace>(ktensor_.ncomponents(), ktensor_.ndims(), sizes_idx);

  if (print)
    std::cout << "Blocking:\n";

  const auto ndims = blocking_.size();
  small_vector<int> grid_pos(ndims, 0);
  for (auto d = 0; d < ndims; ++d) {
    std::vector<int> recvcounts(pmap_->gridSize(), 0);
    std::vector<int> displs(pmap_->gridSize(), 0);
    const auto nblocks = blocking_[d].size() - 1;
    if (print)
      std::cout << "\tDim(" << d << ")\n";
    for (auto b = 0; b < nblocks; ++b) {
      if (print)
        std::cout << "\t\t{" << blocking_[d][b] << ", " << blocking_[d][b + 1]
                  << "} owned by ";
      grid_pos[d] = b;
      int owner = 0;
      MPI_Cart_rank(pmap_->gridComm(), grid_pos.data(), &owner);
      if (print)
        std::cout << owner << "\n";
      recvcounts[owner] =
        ktensor_[d].view().stride(0)*(blocking_[d][b+1]-blocking_[d][b]);
      displs[owner] = ktensor_[d].view().stride(0)*blocking_[d][b];
      grid_pos[d] = 0;
    }

    const bool is_sub_root = pmap_->subCommRank(d) == 0;
    std::size_t send_size = is_sub_root ? ktensor_[d].view().span() : 0;
    MPI_Gatherv(ktensor_[d].view().data(), send_size,
                DistContext::toMpiType<ElementType>(),
                out[d].view().data(), recvcounts.data(), displs.data(),
                DistContext::toMpiType<ElementType>(), 0,
                pmap_->gridComm());
    pmap_->gridBarrier();
  }

  if (print) {
    std::cout << std::endl;
    std::cout << "Subcomm sizes: ";
    for (auto s : pmap_->subCommSizes()) {
      std::cout << s << " ";
    }
    std::cout << std::endl;
  }

  return out;
}

template <typename ElementType, typename ExecSpace>
void
DistKtensor<ElementType, ExecSpace>::
allReduce(const bool divide_by_grid_size)
{
  allReduce(ktensor_, divide_by_grid_size);
}

template <typename ElementType, typename ExecSpace>
void
DistKtensor<ElementType, ExecSpace>::
allReduce(KtensorT<ExecSpace> &u, const bool divide_by_grid_size) const
{
  const ttb_indx nd = u.ndims();
  for (ttb_indx n=0; n<nd; ++n)
    pmap_->subGridAllReduce(
      n, u[n].view().data(), u[n].view().span());

  if (divide_by_grid_size) {
    auto const &gridSizes = pmap_->subCommSizes();
    for (ttb_indx n=0; n<nd; ++n) {
      const ttb_real scale = ttb_real(1.0 / gridSizes[n]);
      u[n].times(scale);
    }
  }
}

template <typename ElementType, typename ExecSpace>
void
DistKtensor<ElementType, ExecSpace>::
export_ktensor(const std::string& file_name) const
{
  KtensorT<ExecSpace> out = importToRoot();
  if (pmap_->gridRank() == 0) {
    // Normalize Ktensor u before writing out
    out.normalize(Genten::NormTwo);
    out.arrange();

    std::cout << "Saving final Ktensor to " << file_name << std::endl;
    auto out_h = create_mirror_view(out);
    deep_copy(out_h, out);
    Genten::export_ktensor(file_name, out_h);
  }
}

template <typename ElementType, typename ExecSpace>
void DistKtensor<ElementType, ExecSpace>::
init_factors(const DistSpTensor<ElementType,ExecSpace>& spTensor)
{
  std::string init_method = input_.get<std::string>("initial-guess", "rand");
  if (init_method == "file") {
    std::string file_name = input_.get<std::string>("initial-file");
    KtensorT<DefaultHostExecutionSpace> u_host;
    import_ktensor(file_name, u_host);
    KtensorT<ExecSpace> u = create_mirror_view(ExecSpace(), u_host);
    deep_copy(u, u_host);
    exportFromRoot(u);
  }
  else if (init_method == "rand") {
    const int seed = input_.get<int>("seed",std::random_device{}());
    const bool prng = input_.get<bool>("prng",true);
    const auto nc = rank();
    const auto nd = ndims();
    const auto norm_x = spTensor.getTensorNorm();
    RandomMT cRMT(seed);
    std::string dist_method =
      input_.get<std::string>("distributed-guess", "serial");
    if (dist_method == "serial") {
      // Compute random ktensor on rank 0 and broadcast to all proc's
      const auto dims = global_dims();
      IndxArrayT<ExecSpace> sz(nd);
      auto hsz = create_mirror_view(sz);
      for (int i=0; i<nd; ++i)
        hsz[i] = dims[i];
      deep_copy(sz,hsz);
      Genten::KtensorT<ExecSpace> u(nc, nd, sz);
      if (pmap_->gridRank() == 0) {
        u.setWeights(1.0);
        u.setMatricesScatter(false, prng, cRMT);
        const auto norm_k = std::sqrt(u.normFsq());
        u.weights().times(norm_x / norm_k);
        u.distribute();
      }
      exportFromRoot(u);
    }
    else if (dist_method == "parallel") {
      // Compute random ktensor on each node
      ktensor_ = KtensorT<ExecSpace>(nc, nd, spTensor.localSpTensor().size());
      ktensor_.setWeights(1.0);
      ktensor_.setMatricesScatter(false, prng, cRMT);

      // Compute global ktensor norm
      ktensor_.setProcessorMap(pmap_.get());
      const auto norm_k = std::sqrt(ktensor_.normFsq());
      ktensor_.setProcessorMap(nullptr);

      ktensor_.weights().times(norm_x / norm_k);
      ktensor_.distribute();
    }
    else if (dist_method == "parallel-drew") {
      // Drew's funky random ktensor that I don't understand
      ktensor_ = KtensorT<ExecSpace>(nc, nd, spTensor.localSpTensor().size());
      ktensor_.setWeights(1.0);
      ktensor_.setMatricesScatter(false, prng, cRMT);
      ktensor_.weights().times(1.0 / norm_x);
      ktensor_.distribute();
      if (pmap_->gridSize() > 1)
        allReduce();
    }
    else
      Genten::error("Unknown distributed-guess method: " + dist_method);
  }
  else
    Genten::error("Unknown initial-guess method: " + init_method);
}

} // namespace Genten
