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

#include "Genten_MixedFormatOps.hpp"
#include "Genten_DistKtensorUpdate.hpp"
#include "Genten_DistKtensorUpdate.hpp"
#include "Genten_TinyVec.hpp"
#include "Genten_AlgParams.hpp"
#include "Genten_SimdKernel.hpp"

namespace Genten {

template <typename row_type, typename val_type, typename ExecSpace>
void row_val_mttkrp(const KtensorT<ExecSpace>& u,
                    const ttb_indx n,
                    const row_type& rows,
                    const val_type& vals,
                    const AlgParams& algParams);

template <typename TensorType>
class DistMttkrp {
public:
  typedef typename TensorType::exec_space exec_space;

private:
  TensorT<exec_space> X_;
  AlgParams algParams_;

public:
  DistMttkrp(const TensorType& X,
             const KtensorT<exec_space>& u,
             const AlgParams& algParams) : X_(X), algParams_(algParams) {}
  ~DistMttkrp() {}

  DistMttkrp(DistMttkrp&&) = delete;
  DistMttkrp(const DistMttkrp&) = delete;
  DistMttkrp& operator=(DistMttkrp&&) = delete;
  DistMttkrp& operator=(const DistMttkrp&) = delete;

  void mttkrp(KtensorT<exec_space>& u,
              const ttb_indx n,
              const FacMatrixT<exec_space>& v) const
  {
    Genten::mttkrp(X_,u,n,v,algParams_);
  }

  void mttkrp_all(KtensorT<exec_space>& u,
                  const KtensorT<exec_space>& v) const
  {
    Genten::mttkrp_all(X_, u, v, algParams_);
  }
};

template <typename ExecSpace>
class DistMttkrp<SptensorT<ExecSpace> > {
public:
  typedef ExecSpace exec_space;

private:
  SptensorT<exec_space> X_;
  AlgParams algParams_;
  DistKtensorUpdate<exec_space>* distUpdate_;

public:

  DistMttkrp(const SptensorT<exec_space>& X,
             const KtensorT<exec_space>& u,
             const ttb_indx nnz,
             const AlgParams& algParams) :
    X_(X), algParams_(algParams), distUpdate_(nullptr)
  {
    if (algParams_.dist_update_method == Dist_Update_Method::AllReduce)
      distUpdate_ = new KtensorAllReduceUpdate<exec_space>(u);
    else if (algParams_.dist_update_method == Dist_Update_Method::AllGather) {
      distUpdate_ = new KtensorAllGatherUpdate<exec_space>(u, nnz);
    }
    else
      Genten::error("Unknown distributed update method");
  }

  DistMttkrp(const SptensorT<exec_space>& X,
             const KtensorT<exec_space>& u,
             const AlgParams& algParams) :
    DistMttkrp(X, u, X.nnz(), algParams) {}

  ~DistMttkrp()
  {
    if (distUpdate_ != nullptr)
      delete distUpdate_;
  }

  DistMttkrp(DistMttkrp&&) = delete;
  DistMttkrp(const DistMttkrp&) = delete;
  DistMttkrp& operator=(DistMttkrp&&) = delete;
  DistMttkrp& operator=(const DistMttkrp&) = delete;

  void mttkrp(KtensorT<exec_space>& u,
              const ttb_indx n,
              const FacMatrixT<exec_space>& v) const
  {
    assert(distUpdate_ != nullptr);

    auto pmap = u.getProcessorMap();
    u.setProcessorMap(nullptr);

    if (algParams_.dist_update_method == Dist_Update_Method::AllGather) {
      KtensorAllGatherUpdate<exec_space>* allGatherUpdate = dynamic_cast<KtensorAllGatherUpdate<exec_space>*>(distUpdate_);
      assert(allGatherUpdate != nullptr);

      auto rows = allGatherUpdate->getRowUpdates(n);
      auto vals = allGatherUpdate->getValUpdates(n);
      row_val_mttkrp(X_, u, n, rows, vals, algParams_);
    }
    else
      Genten::mttkrp(X_, u, n, v, algParams_);

    distUpdate_->update(v,n);

    u.setProcessorMap(pmap);
  }

  void mttkrp_all(KtensorT<exec_space>& u,
                  const KtensorT<exec_space>& v) const
  {
    assert(distUpdate_ != nullptr);

    auto pmap = u.getProcessorMap();
    u.setProcessorMap(nullptr);

    if (algParams_.dist_update_method == Dist_Update_Method::AllGather) {
      const ttb_indx nd = u.ndims();
      for (ttb_indx n=0; n<nd; ++n)
        mttkrp(u, n, v[n]);
    }
    else
      Genten::mttkrp_all(X_, u, v, algParams_);

    distUpdate_->update(v);

    u.setProcessorMap(pmap);
  }

};

namespace Impl {
template <unsigned FBS, unsigned VS, typename row_type, typename val_type,
           typename ExecSpace>
void
row_val_mttkrp_kernel(const SptensorT<ExecSpace>& X,
                      const KtensorT<ExecSpace>& u,
                      const unsigned n,
                      const row_type& rows,
                      const val_type& vals,
                      const AlgParams& algParams)
{
  static const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;
  static const unsigned FacBlockSize = FBS;
  static const unsigned VectorSize = is_gpu ? VS : 1;
  static const unsigned TeamSize = is_gpu ? 128/VectorSize : 1;
  /*const*/ unsigned RowBlockSize = algParams.mttkrp_nnz_tile_size;
  const unsigned RowsPerTeam = TeamSize * RowBlockSize;

  /*const*/ unsigned nd = u.ndims();
  /*const*/ unsigned nc_total = u.ncomponents();
  /*const*/ ttb_indx nnz = X.nnz();
  const ttb_indx N = (nnz+RowsPerTeam-1)/RowsPerTeam;

  typedef Kokkos::TeamPolicy<ExecSpace> Policy;
  typedef typename Policy::member_type TeamMember;
  Policy policy(N, TeamSize, VectorSize);

  // Use factor matrix tile size as requested by the user, or all columns if
  // unspecified
  const unsigned FacTileSize =
    algParams.mttkrp_duplicated_factor_matrix_tile_size > 0 ? algParams.mttkrp_duplicated_factor_matrix_tile_size : nc_total;
  for (unsigned nc_beg=0; nc_beg<nc_total; nc_beg += FacTileSize) {
    const unsigned nc =
      nc_beg+FacTileSize <= nc_total ? FacTileSize : nc_total-nc_beg;
    const unsigned nc_end = nc_beg+nc;
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const TeamMember& team)
    {
      ttb_indx offset =
        (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
      ttb_indx stride = 1;

      auto row_func = [&](auto j, auto nj, auto Nj) {
        typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TVM;
        for (unsigned ii=0; ii<RowBlockSize; ++ii) {
          const ttb_indx i = offset + ii*stride;
          if (i >= nnz)
            continue;

          const ttb_indx k = X.subscript(i,n);
          const ttb_real x_val = X.value(i);

          // MTTKRP for dimension n
          auto tmp = TVM::make(team, nj, x_val);
          tmp *= &(u.weights(nc_beg+j));
          for (unsigned m=0; m<nd; ++m) {
            if (m != n)
              tmp *= &(u[m].entry(X.subscript(i,m),nc_beg+j));
          }
          tmp.store(&(vals(i,j)));
          rows(i) = k;
        }
      };

      for (unsigned j=0; j<nc; j+=FacBlockSize) {
        if (j+FacBlockSize <= nc) {
          const unsigned nj = FacBlockSize;
          row_func(j, nj, std::integral_constant<unsigned,nj>());
        }
        else {
          const unsigned nj = nc-j;
          row_func(j, nj, std::integral_constant<unsigned,0>());
        }
      }
    }, "row_val_mttkrp_kernel");
  }
}

template <typename RowType, typename ValType, typename ExecSpace>
struct Row_Val_MTTKRP_Kernel {
  const SptensorT<ExecSpace> X;
  const KtensorT<ExecSpace> u;
  const ttb_indx n;
  const RowType rows;
  const ValType vals;
  const AlgParams algParams;

  Row_Val_MTTKRP_Kernel(const SptensorT<ExecSpace>& X_,
                        const KtensorT<ExecSpace>& u_,
                        const ttb_indx n_,
                        const RowType& rows_,
                        const ValType& vals_,
                        const AlgParams& algParams_) :
    X(X_), u(u_), n(n_), rows(rows_), vals(vals_), algParams(algParams_) {}

  template <unsigned FBS, unsigned VS>
  void run() const {
      row_val_mttkrp_kernel<FBS,VS>(
        X,u,n,rows,vals,algParams);
  }
};

}

template <typename row_type, typename val_type, typename ExecSpace>
void row_val_mttkrp(const SptensorT<ExecSpace>& X,
                    const KtensorT<ExecSpace>& u,
                    const ttb_indx n,
                    const row_type& rows,
                    const val_type& vals,
                    const AlgParams& algParams)
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::mttkrp");
#endif

  const ttb_indx nc = u.ncomponents();     // Number of components
  const ttb_indx nd = u.ndims();           // Number of dimensions

  assert(X.ndims() == nd);
  assert(u.isConsistent());
  for (ttb_indx i = 0; i < nd; i++)
  {
    if (i != n)
      assert(u[i].nRows() == X.size(i));
  }
  assert( rows.extent(0) == X.nnz() );
  assert( vals.extent(0) == X.nnz() );

  Impl::Row_Val_MTTKRP_Kernel<row_type,val_type,ExecSpace>
    kernel(X,u,n,rows,vals,algParams);
  Impl::run_row_simd_kernel(kernel, nc);
}

}
