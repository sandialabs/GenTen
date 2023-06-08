//@HEADER
// ************************************************************************
//     C++ Tensor Toolbox
//     Software package for tensor math by Sandia National Laboratories
//
// Sandia National Laboratories is a multiprogram laboratory operated by
// Sandia Corporation, a wholly owned subsidiary of Lockheed Martin Corporation,
// for the United States Department of Energy's National Nuclear Security
// Administration under contract DE-AC04-94AL85000.
//
// Copyright 2013, Sandia Corporation.
// ************************************************************************
//@HEADER

#include "Genten_Tensor.hpp"
#include "Genten_Sptensor.hpp"

namespace Genten {

namespace Impl {

template <typename ExecSpace, typename Layout>
void copyFromSptensor(const TensorImpl<ExecSpace,Layout>& x,
                      const SptensorImpl<ExecSpace>& src)
{
  const ttb_indx nnz = src.nnz();
  Kokkos::parallel_for("copyFromSptensor",
                       Kokkos::RangePolicy<ExecSpace>(0,nnz),
                       KOKKOS_LAMBDA(const ttb_indx i)
  {
    const ttb_indx k = x.sub2ind(src.getSubscripts(i));
    x[k] = src.value(i);
  });
}

template <typename ExecSpace, typename Layout>
void copyFromKtensor(const TensorImpl<ExecSpace,Layout>& x,
                     const KtensorImpl<ExecSpace>& src)
{
  typedef Kokkos::TeamPolicy<ExecSpace> Policy;
  typedef typename Policy::member_type TeamMember;
  typedef Kokkos::View< ttb_indx**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

  /*const*/ ttb_indx ne = x.numel();
  /*const*/ unsigned nd = src.ndims();
  /*const*/ unsigned nc = src.ncomponents();

  // Make VectorSize*TeamSize ~= 256 on Cuda and HIP
  static const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;
  const unsigned VectorSize = is_gpu ? nc : 1;
  const unsigned TeamSize = is_gpu ? (256+nc-1)/nc : 1;
  const ttb_indx N = (ne+TeamSize-1)/TeamSize;

  const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,nd);
  Policy policy(N, TeamSize, VectorSize);
  Kokkos::parallel_for("copyFromKtensor",
                       policy.set_scratch_size(0,Kokkos::PerTeam(bytes)),
                       KOKKOS_LAMBDA(const TeamMember& team)
  {
    // Compute indices for entry "i"
    const unsigned team_rank = team.team_rank();
    const unsigned team_size = team.team_size();
    const ttb_indx i = team.league_rank()*team_size+team_rank;
    if (i >= ne)
      return;

    TmpScratchSpace scratch(team.team_scratch(0), team_size, nd);
    ttb_indx *sub = &scratch(team_rank, 0);
    Kokkos::single(Kokkos::PerThread(team), [&]()
    {
      x.ind2sub(sub, i);
    });

    // Compute Ktensor value for given indices
    ttb_real src_val = 0.0;
    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, nc),
                            [&](const unsigned j, ttb_real& v)
    {
      ttb_real tmp = src.weights(j);
      for (unsigned m=0; m<nd; ++m) {
        tmp *= src[m].entry(sub[m],j);
      }
      v += tmp;
    }, src_val);

    // Write result
    Kokkos::single(Kokkos::PerThread(team), [&]()
    {
      x[i] = src_val;
    });

  });
}

}

template <typename ExecSpace, typename Layout>
TensorImpl<ExecSpace,Layout>::
TensorImpl(const SptensorT<ExecSpace>& src) :
  siz(src.size()),
  lower_bound(src.getLowerBounds().clone()),
  upper_bound(src.getUpperBounds().clone())
{
  siz_host = create_mirror_view(siz);
  deep_copy(siz_host, siz);
  values = ArrayT<ExecSpace>(siz_host.prod(), ttb_real(0.0));
  Impl::copyFromSptensor(*this, src.impl());
}

template <typename ExecSpace, typename Layout>
TensorImpl<ExecSpace,Layout>::
TensorImpl(const KtensorT<ExecSpace>& src) :
  siz(src.ndims()),
  lower_bound(src.ndims(),ttb_indx(0))
{
  siz_host = create_mirror_view(siz);
  const ttb_indx nd = siz_host.size();
  for (ttb_indx i=0; i<nd; ++i)
    siz_host[i] = src[i].nRows();
  deep_copy(siz, siz_host);
  values = ArrayT<ExecSpace>(siz_host.prod());
  Impl::copyFromKtensor(*this, src.impl());
  upper_bound = siz.clone();
}

namespace Impl {

template <typename LayoutTrans, typename ExecSpace, typename Layout>
TensorImpl<ExecSpace,LayoutTrans>
transpose(const TensorImpl<ExecSpace,Layout>& X)
{
  typedef Kokkos::TeamPolicy<ExecSpace> Policy;
  typedef typename Policy::member_type TeamMember;
  typedef Kokkos::View< ttb_indx**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

  const ttb_indx nd = X.ndims();
  const ttb_indx ne = X.numel();
  const Genten::IndxArray sz_host = X.size_host();
  Genten::IndxArrayT<ExecSpace> szt(nd);
  auto szt_host = create_mirror_view(szt);
  for (ttb_indx i=0; i<nd; ++i)
    szt_host[i] = sz_host[nd-i-1];
  deep_copy(szt, szt_host);
  TensorImpl<ExecSpace,LayoutTrans> Xt(szt);
  deep_copy(Xt.getLowerBounds(), X.getLowerBounds());
  deep_copy(Xt.getUpperBounds(), X.getUpperBounds());

  static const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;
  const unsigned VectorSize = 1;
  const unsigned TeamSize = is_gpu ? 32 : 1;
  const ttb_indx N = (ne+TeamSize-1)/TeamSize;

  const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,2*nd);
  Policy policy(N, TeamSize, VectorSize);
  Kokkos::parallel_for("Tensor::transpose",
                       policy.set_scratch_size(0,Kokkos::PerTeam(bytes)),
                       KOKKOS_LAMBDA(const TeamMember& team)
  {
    // Compute indices for entry "i"
    const unsigned team_rank = team.team_rank();
    const unsigned team_size = team.team_size();
    const ttb_indx i = team.league_rank()*team_size+team_rank;
    if (i >= ne)
      return;

    TmpScratchSpace scratch(team.team_scratch(0), team_size, 2*nd);
    ttb_indx *s = &scratch(team_rank, 0);
    ttb_indx *st = &scratch(team_rank, nd);

    Kokkos::single(Kokkos::PerThread(team), [&]()
    {
      // Map linearized index to multi-index
      X.ind2sub(s, i);

      // Compute multi-index of transposed tensor
      for (ttb_indx j=0; j<nd; ++j)
        st[j] = s[nd-j-1];

      // Map transpose multi-index to transposed linearized index
      const ttb_indx k = Xt.sub2ind(st);

      Xt[k] = X[i];
    });
  });

  return Xt;
}

template <typename LayoutTrans, typename ExecSpace, typename Layout>
TensorImpl<ExecSpace,LayoutTrans>
switch_layout(const TensorImpl<ExecSpace,Layout>& X)
{
  typedef Kokkos::TeamPolicy<ExecSpace> Policy;
  typedef typename Policy::member_type TeamMember;
  typedef Kokkos::View< ttb_indx**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

  const ttb_indx nd = X.ndims();
  const ttb_indx ne = X.numel();
  TensorImpl<ExecSpace,LayoutTrans> X2(X.size());
  deep_copy(X2.getLowerBounds(), X.getLowerBounds());
  deep_copy(X2.getUpperBounds(), X.getUpperBounds());

  static const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;
  const unsigned VectorSize = 1;
  const unsigned TeamSize = is_gpu ? 32 : 1;
  const ttb_indx N = (ne+TeamSize-1)/TeamSize;

  const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,nd);
  Policy policy(N, TeamSize, VectorSize);
  Kokkos::parallel_for("Tensor::switch_layout",
                       policy.set_scratch_size(0,Kokkos::PerTeam(bytes)),
                       KOKKOS_LAMBDA(const TeamMember& team)
  {
    // Compute indices for entry "i"
    const unsigned team_rank = team.team_rank();
    const unsigned team_size = team.team_size();
    const ttb_indx i = team.league_rank()*team_size+team_rank;
    if (i >= ne)
      return;

    TmpScratchSpace scratch(team.team_scratch(0), team_size, nd);
    ttb_indx *s = &scratch(team_rank, 0);

    Kokkos::single(Kokkos::PerThread(team), [&]()
    {
      // Map linearized index to multi-index
      X.ind2sub(s, i);

      // Map multi-index to new linearized index
      const ttb_indx k = X2.sub2ind(s);

      X2[k] = X[i];
    });
  });

  return X2;
}

}

template <typename ExecSpace>
TensorT<ExecSpace>
TensorT<ExecSpace>::
transpose(TensorLayout new_layout) const
{
  TensorT<ExecSpace> X;
  if (has_left_impl()) {
    auto impl = left_impl();
    if (new_layout == TensorLayout::Left)
      X = TensorT<ExecSpace>(Impl::transpose<Impl::TensorLayoutLeft>(impl));
    else
      X = TensorT<ExecSpace>(Impl::transpose<Impl::TensorLayoutRight>(impl));
  }
  else {
    auto impl = right_impl();
    if (new_layout == TensorLayout::Left)
      X = TensorT<ExecSpace>(Impl::transpose<Impl::TensorLayoutLeft>(impl));
    else
      X = TensorT<ExecSpace>(Impl::transpose<Impl::TensorLayoutRight>(impl));
  }
  return X;
}

template <typename ExecSpace>
TensorT<ExecSpace>
TensorT<ExecSpace>::
switch_layout(TensorLayout new_layout) const
{
  TensorT<ExecSpace> X;
  if (has_left_impl()) {
    auto impl = left_impl();
    if (new_layout == TensorLayout::Left)
      X = TensorT<ExecSpace>(Impl::switch_layout<Impl::TensorLayoutLeft>(impl));
    else
      X = TensorT<ExecSpace>(Impl::switch_layout<Impl::TensorLayoutRight>(impl));
  }
  else {
    auto impl = right_impl();
    if (new_layout == TensorLayout::Left)
      X = TensorT<ExecSpace>(Impl::switch_layout<Impl::TensorLayoutLeft>(impl));
    else
      X = TensorT<ExecSpace>(Impl::switch_layout<Impl::TensorLayoutRight>(impl));
  }
  return X;
}

}

#define INST_MACRO(SPACE) \
  template class Genten::TensorImpl<SPACE,Genten::Impl::TensorLayoutLeft>; \
  template class Genten::TensorImpl<SPACE,Genten::Impl::TensorLayoutRight>; \
  template class Genten::TensorT<SPACE>;
GENTEN_INST(INST_MACRO)
