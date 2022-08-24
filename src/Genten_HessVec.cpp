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

#include "Genten_HessVec.hpp"

#include <assert.h>
#include <type_traits>

#include "Genten_Util.hpp"
#include "Genten_FacMatrix.hpp"
#include "Genten_TinyVec.hpp"
#include "Genten_SimdKernel.hpp"

// This is a locally-modified version of Kokkos_ScatterView.hpp which we
// need until the changes are moved into Kokkos
#include "Genten_Kokkos_ScatterView.hpp"

#ifdef HAVE_CALIPER
#include <caliper/cali.h>
#endif

//-----------------------------------------------------------------------------
//  Method:  hessvec, Sptensor X, output to FacMatrix
//-----------------------------------------------------------------------------

namespace Genten {
namespace Impl {

// HessVec kernel for Sptensor for all modes simultaneously
// Because of problems with ScatterView, doesn't work on the GPU
template <int Dupl, int Cont, typename ExecSpace>
struct HessVec_Kernel {
  const SptensorT<ExecSpace> XX;
  const KtensorT<ExecSpace> aa;
  const KtensorT<ExecSpace> vv;
  const KtensorT<ExecSpace> uu;
  const AlgParams algParams;

  HessVec_Kernel(const SptensorT<ExecSpace>& X_,
                 const KtensorT<ExecSpace>& a_,
                 const KtensorT<ExecSpace>& v_,
                 const KtensorT<ExecSpace>& u_,
                 const AlgParams& algParams_) :
    XX(X_), aa(a_), vv(v_), uu(u_), algParams(algParams_) {}

  template <unsigned FBS, unsigned VS>
  void run() const {
    const SptensorT<ExecSpace> X = XX;
    const KtensorT<ExecSpace> a = aa;
    const KtensorT<ExecSpace> v = vv;
    const KtensorT<ExecSpace> u = uu;

    u.setMatrices(0.0);

    using Kokkos::Experimental::create_scatter_view;
    using Kokkos::Experimental::ScatterView;
    using Kokkos::Experimental::ScatterSum;

    static const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;
    static const unsigned FacBlockSize = FBS;
    static const unsigned VectorSize = is_gpu ? VS : 1;
    static const unsigned TeamSize = is_gpu ? 128/VectorSize : 1;
    /*const*/ unsigned RowBlockSize = algParams.mttkrp_nnz_tile_size;
    const unsigned RowsPerTeam = TeamSize * RowBlockSize;

    static_assert(!is_gpu, "Cannot call hessvec_kernel for Cuda, HIP or SYCL space!");

    /*const*/ unsigned nd = a.ndims();
    /*const*/ unsigned nc_total = a.ncomponents();
    /*const*/ ttb_indx nnz = X.nnz();
    const ttb_indx N = (nnz+RowsPerTeam-1)/RowsPerTeam;

    typedef Kokkos::TeamPolicy<ExecSpace> Policy;
    typedef typename Policy::member_type TeamMember;
    Policy policy(N, TeamSize, VectorSize);

    typedef ScatterView<ttb_real**,Kokkos::LayoutRight,ExecSpace,ScatterSum,Dupl,Cont> ScatterViewType;

    // Use factor matrix tile size as requested by the user, or all columns if
    // unspecified
    const unsigned FacTileSize =
      algParams.mttkrp_duplicated_factor_matrix_tile_size > 0 ? algParams.mttkrp_duplicated_factor_matrix_tile_size : nc_total;
    for (unsigned nc_beg=0; nc_beg<nc_total; nc_beg += FacTileSize) {
      const unsigned nc =
        nc_beg+FacTileSize <= nc_total ? FacTileSize : nc_total-nc_beg;
      const unsigned nc_end = nc_beg+nc;
      ScatterViewType *su = new ScatterViewType[nd];
      for (unsigned n=0; n<nd; ++n) {
        auto uu = Kokkos::subview(u[n].view(),Kokkos::ALL,
                                  std::make_pair(nc_beg,nc_end));
        su[n] = ScatterViewType(uu);
      }
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const TeamMember& team)
      {
        // Loop over tensor non-zeros with a large stride on the GPU to
        // reduce atomic contention when the non-zeros are in a nearly sorted
        // order (often the first dimension of the tensor).  This is similar
        // to an approach used in ParTi (https://github.com/hpcgarage/ParTI)
        // by Jaijai Li.
        ttb_indx offset;
        ttb_indx stride;
        if (is_gpu) {
          offset = team.league_rank()*TeamSize+team.team_rank();
          stride = team.league_size()*TeamSize;
        }
        else {
          offset =
            (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
          stride = 1;
        }

        auto row_func = [&](auto j, auto nj, auto Nj) {
          typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TVM;
          for (unsigned ll=0; ll<RowBlockSize; ++ll) {
            const ttb_indx l = offset + ll*stride;
            if (l >= nnz)
              continue;

            const ttb_real x_val = X.value(l);

            for (unsigned k=0; k<nd; ++k) {
              const ttb_indx i = X.subscript(l,k);
              auto vu = su[k].access();
              auto tmp2 = TVM::make(team, nj, 0.0);
              for (unsigned s=0; s<nd; ++s) {
                if (s != k) {
                  auto tmp = TVM::make(team, nj, x_val);
                  tmp *= &(a.weights(nc_beg+j));
                  for (unsigned n=0; n<nd; ++n) {
                    if (n != k && n != s)
                      tmp *= &(a[n].entry(X.subscript(l,n),nc_beg+j));
                  }
                  tmp *= &(v[s].entry(X.subscript(l,s),nc_beg+j));
                  tmp2 += tmp;
                }
              }
              vu(i,j) += tmp2;
            }
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
      }, "hessvec_kernel");

      for (unsigned n=0; n<nd; ++n) {
        auto uu = Kokkos::subview(u[n].view(),Kokkos::ALL,
                                  std::make_pair(nc_beg,nc_end));
        su[n].contribute_into(uu);
      }
      delete [] su;
    }
  }
};

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(ENABLE_SYCL_FOR_CUDA)
// Specialization for Cuda, HIP or SYCL that always uses atomics and doesn't call
// mttkrp_all_kernel, which won't run on the GPU
template <int Dupl, int Cont>
struct HessVec_Kernel<Dupl, Cont, Kokkos_GPU_Space> {
  typedef Kokkos_GPU_Space ExecSpace;

  const SptensorT<ExecSpace> XX;
  const KtensorT<ExecSpace> aa;
  const KtensorT<ExecSpace> vv;
  const KtensorT<ExecSpace> uu;
  const AlgParams algParams;

  HessVec_Kernel(const SptensorT<ExecSpace>& X_,
                 const KtensorT<ExecSpace>& a_,
                 const KtensorT<ExecSpace>& v_,
                 const KtensorT<ExecSpace>& u_,
                 const AlgParams& algParams_) :
    XX(X_), aa(a_), vv(v_), uu(u_), algParams(algParams_) {}

  template <unsigned FBS, unsigned VS>
  void run() const {
    const SptensorT<ExecSpace> X = XX;
    const KtensorT<ExecSpace> a = aa;
    const KtensorT<ExecSpace> v = vv;
    const KtensorT<ExecSpace> u = uu;

    if (algParams.mttkrp_all_method != MTTKRP_All_Method::Atomic)
      Genten::error("MTTKRP-All method must be atomic on Cuda, HIP or SYCL!");

    u.setMatrices(0.0);

    static const unsigned FacBlockSize = FBS;
    static const unsigned VectorSize = VS;
    static const unsigned TeamSize = 128/VectorSize;
    /*const*/ unsigned RowBlockSize = algParams.mttkrp_nnz_tile_size;
    const unsigned RowsPerTeam = TeamSize * RowBlockSize;

    /*const*/ unsigned nd = a.ndims();
    /*const*/ unsigned nc = a.ncomponents();
    /*const*/ ttb_indx nnz = X.nnz();
    const ttb_indx N = (nnz+RowsPerTeam-1)/RowsPerTeam;

    typedef Kokkos::TeamPolicy<ExecSpace> Policy;
    typedef typename Policy::member_type TeamMember;
    Policy policy(N, TeamSize, VectorSize);

    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const TeamMember& team)
    {
      // Loop over tensor non-zeros with a large stride on the GPU to
      // reduce atomic contention when the non-zeros are in a nearly sorted
      // order (often the first dimension of the tensor).  This is similar
      // to an approach used in ParTi (https://github.com/hpcgarage/ParTI)
      // by Jaijai Li.
      ttb_indx offset = team.league_rank()*TeamSize+team.team_rank();
      ttb_indx stride = team.league_size()*TeamSize;

      auto row_func = [&](auto j, auto nj, auto Nj) {
        typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TVM;
        for (unsigned ll=0; ll<RowBlockSize; ++ll) {
          const ttb_indx l = offset + ll*stride;
          if (l >= nnz)
            continue;

          const ttb_real x_val = X.value(l);

          for (unsigned k=0; k<nd; ++k) {
            const ttb_indx i = X.subscript(l,k);
            auto tmp2 = TVM::make(team, nj, 0.0);
            for (unsigned s=0; s<nd; ++s) {
              if (s != k) {
                auto tmp = TVM::make(team, nj, x_val);
                tmp *= &(a.weights(j));
                for (unsigned n=0; n<nd; ++n) {
                  if (n != k && n != s)
                    tmp *= &(a[n].entry(X.subscript(l,n),j));
                }
                tmp *= &(v[s].entry(X.subscript(l,s),j));
                tmp2 += tmp;
              }
            }
            Kokkos::atomic_add(&u[k].entry(i,j), tmp2);
          }
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
    }, "hessvec_kernel");
  }
};
#endif

// HessVec permutation-based kernel for Sptensor for all modes simultaneously
template <typename ExecSpace>
struct HessVec_PermKernel {
  const SptensorT<ExecSpace> XX;
  const KtensorT<ExecSpace> aa;
  const KtensorT<ExecSpace> vv;
  const KtensorT<ExecSpace> uu;
  const AlgParams algParams;

  HessVec_PermKernel(const SptensorT<ExecSpace>& X_,
                     const KtensorT<ExecSpace>& a_,
                     const KtensorT<ExecSpace>& v_,
                     const KtensorT<ExecSpace>& u_,
                     const AlgParams& algParams_) :
    XX(X_), aa(a_), vv(v_), uu(u_), algParams(algParams_) {}

  template <unsigned FBS, unsigned VS>
  void run() const {
    const SptensorT<ExecSpace> X = XX;
    const KtensorT<ExecSpace> a = aa;
    const KtensorT<ExecSpace> v = vv;
    const KtensorT<ExecSpace> u = uu;

    u.setMatrices(0.0);

    static const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;
    static const unsigned FacBlockSize = FBS;
    static const unsigned VectorSize = is_gpu ? VS : 1;
    static const unsigned TeamSize = is_gpu ? 128/VectorSize : 1;
    /*const*/ unsigned RowBlockSize = algParams.mttkrp_nnz_tile_size;
    const unsigned RowsPerTeam = TeamSize * RowBlockSize;

    /*const*/ unsigned nd = a.ndims();
    /*const*/ unsigned nc = a.ncomponents();
    /*const*/ ttb_indx nnz = X.nnz();
    const ttb_indx N = (nnz+RowsPerTeam-1)/RowsPerTeam;

    typedef Kokkos::TeamPolicy<ExecSpace> Policy;
    typedef typename Policy::member_type TeamMember;
    Policy policy(N, TeamSize, VectorSize);

    // Perm only works for a single dimension at a time, so loop over them
    // outside the kernel
    for (unsigned k=0; k<nd; ++k) {
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const TeamMember& team)
      {
        /*const*/ ttb_indx invalid_row = ttb_indx(-1);
        /*const*/ ttb_indx i_block =
          (team.league_rank()*TeamSize + team.team_rank())*RowBlockSize;

        auto row_func = [&](auto j, auto nj, auto Nj) {
          typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TVM;
          auto val = TVM::make(team, nj, 0.0);
          auto tmp = TVM::make(team, nj, 0.0);

          ttb_indx row_prev = invalid_row;
          ttb_indx row = invalid_row;
          ttb_indx first_row = invalid_row;
          ttb_indx p = invalid_row;
          ttb_real x_val = 0.0;

          for (unsigned ii=0; ii<RowBlockSize; ++ii) {
            /*const*/ ttb_indx i = i_block+ii;

            if (i >= nnz)
              row = invalid_row;
            else {
              p = X.getPerm(i,k);
              x_val = X.value(p);
              row = X.subscript(p,k);
            }

            if (ii == 0)
              first_row = row;

            // If we got a different row index, add in result
            if (row != row_prev) {
              if (row_prev != invalid_row) {
                if (row_prev == first_row) // Only need atomics for first/last row
                  Kokkos::atomic_add(&u[k].entry(row_prev,j), val);
                else
                  val.store_plus(&u[k].entry(row_prev,j));
                val.broadcast(0.0);
              }
              row_prev = row;
            }

            if (row != invalid_row) {
              for (unsigned s=0; s<nd; ++s) {
                if (s != k) {
                  tmp.load(&(a.weights(j)));
                  tmp *= x_val;
                  for (unsigned n=0; n<nd; ++n) {
                    if (n != k && n != s)
                      tmp *= &(a[n].entry(X.subscript(p,n),j));
                  }
                  tmp *= &(v[s].entry(X.subscript(p,s),j));
                  val += tmp;
                }
              }
            }
          }

          // Sum in last row
          if (row != invalid_row) {
            Kokkos::atomic_add(&u[k].entry(row,j), val);
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
      }, "hessvec_perm_kernel");
    }
  }
};

// This is a very poor implementation of hess_vec for dense tensors, using
// the same (bad) parallelism strategy as for dense MTTKRP
template <typename ExecSpace>
struct HessVec_Dense_Kernel {
  const TensorT<ExecSpace> XX;
  const KtensorT<ExecSpace> aa;
  const KtensorT<ExecSpace> vv;
  const KtensorT<ExecSpace> uu;
  const AlgParams algParams;

  HessVec_Dense_Kernel(const TensorT<ExecSpace>& X_,
                       const KtensorT<ExecSpace>& a_,
                       const KtensorT<ExecSpace>& v_,
                       const KtensorT<ExecSpace>& u_,
                       const AlgParams& algParams_) :
    XX(X_), aa(a_), vv(v_), uu(u_), algParams(algParams_) {}

  template <unsigned FBS, unsigned VS>
  void run() const {
    const TensorT<ExecSpace> X = XX;
    const KtensorT<ExecSpace> a = aa;
    const KtensorT<ExecSpace> v = vv;
    const KtensorT<ExecSpace> u = uu;

    typedef Kokkos::TeamPolicy<ExecSpace> Policy;
    typedef typename Policy::member_type TeamMember;
    typedef Kokkos::View< ttb_indx**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

    u.setMatrices(0.0);

    static const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;
    static const unsigned FacBlockSize = FBS;
    static const unsigned VectorSize = is_gpu ? VS : 1;
    static const unsigned TeamSize = is_gpu ? 128/VectorSize : 1;

    /*const*/ unsigned nd = a.ndims();
    /*const*/ unsigned nc = a.ncomponents();
    const size_t bytes = TmpScratchSpace::shmem_size(TeamSize, nd);

    for (unsigned k=0; k<nd; ++k) {
      /*const*/ ttb_indx ns = X.size(k);
      const ttb_indx N = (ns+TeamSize-1)/TeamSize;
      Policy policy(N, TeamSize, VectorSize);
      Kokkos::parallel_for(policy.set_scratch_size(0,Kokkos::PerTeam(bytes)),
                           KOKKOS_LAMBDA(const TeamMember& team)
      {
        // Row of u we write to
        const unsigned team_rank = team.team_rank();
        const unsigned team_size = team.team_size();
        /*const*/ ttb_indx i = team.league_rank()*team_size + team_rank;
        if (i >= ns)
          return;

        // Scratch space for storing tensor subscripts
        TmpScratchSpace scratch(team.team_scratch(0), team_size, nd);
        ttb_indx *sub = &scratch(team_rank, 0);

        auto row_func = [&](auto j, auto nj, auto Nj) {
          typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TVM;

          // Work around internal-compiler errors in recent Intel compilers
          unsigned nd_ = nd;
          unsigned k_ = k;
          TensorT<ExecSpace> X_ = X;

          // Initialize our subscript array for row i of mode n
          Kokkos::single(Kokkos::PerThread(team), [&]()
          {
            for (unsigned l=0; l<nd_; ++l)
              sub[l] = 0;
            sub[k_] = i;
          });

          auto val = TVM::make(team, nj, 0.0);
          int done = 0;
          while (!done) {  // Would like to get some parallelism in this loop
            const ttb_indx l = X.sub2ind(sub);
            const ttb_real x_val = X[l];
            auto tmp2 = TVM::make(team, nj, 0.0);
            for (unsigned s=0; s<nd; ++s) {
              if (s != k) {
                auto tmp = TVM::make(team, nj, x_val);
                tmp *= &(a.weights(j));
                for (unsigned n=0; n<nd; ++n) {
                  if (n != k && n != s)
                    tmp *= &(a[n].entry(sub[n],j));
                }
                tmp *= &(v[s].entry(sub[s],j));
                tmp2 += tmp;
              }
            }
            val += tmp2;

            Kokkos::single(Kokkos::PerThread(team), [&](int& dn)
            {
              dn = !X_.increment_sub(sub,k_);
            }, done);
          }
          val.store_plus(&u[k].entry(i,j));
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
      }, "hessvec_kernel");
    }
  }
};

// This computes the ktensor-only contribution to the Hessian, which
// doesn't depend on the tensor
template <typename ExecSpace>
void hess_vec_ktensor_term(const KtensorT<ExecSpace>& a,
                           const KtensorT<ExecSpace>& v,
                           const KtensorT<ExecSpace>& u)
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::hess_vec_ktensor_term");
#endif

  const ProcessorMap *pmap = u.getProcessorMap();
  const ttb_indx nc = a.ncomponents();     // Number of components
  const ttb_indx nd = a.ndims();           // Number of dimensions
  IndxArrayT<ExecSpace> nrow(nd, nc);
  FacMatArrayT<ExecSpace> Z(nd, nrow, nc), W(nd, nrow, nc);
  for (unsigned n=0; n<nd; ++n) {
    Z[n].gramian(a[n], true); // full gram
    W[n].gemm(true, false, 1.0, a[n], v[n], 0.0);  // a[n]^T * v[n]
    if (pmap != nullptr)
      pmap->facMap(n)->allReduce(W[n].view().data(), W[n].view().span());
  }

  for (unsigned k=0; k<nd; ++k) {
    const ttb_indx I_k = a[k].nRows();
    Kokkos::RangePolicy<ExecSpace> policy(0,I_k);
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ttb_indx i)
    {
      for (unsigned j=0; j<nc; ++j) {
        for (unsigned s=0; s<nd; ++s) {
          const ttb_indx I_s = a[s].nRows();
          if (s == k) {
            for (unsigned p=0; p<nc; ++p) {
              ttb_real tmp = 1.0;
              for (unsigned n=0; n<nd; ++n) {
                if (n != k)
                  tmp *= Z[n].entry(p,j);
              } // n
              u[k].entry(i,j) += tmp * v[s].entry(i,p);
            } // p
          }
          else {
            for (unsigned p=0; p<nc; ++p) {
              if (p != j) {
                ttb_real tmp = a[k].entry(i,p)*W[s].entry(j,p);
                for (unsigned n=0; n<nd; ++n) {
                  if (n != k && n != s)
                    tmp *= Z[n].entry(p,j);
                }
                u[k].entry(i,j) += tmp;
              }
              else {
                ttb_real tmp2 = 0.0;
                for (unsigned q=0; q<nc; ++q) {
                  ttb_real tmp = a[k].entry(i,q)*W[s].entry(q,j);
                  if (q == j)
                    tmp *= 2.0;
                  for (unsigned n=0; n<nd; ++n) {
                    if (n != k && n != s)
                      tmp *= Z[n].entry(q,j);
                  } // n
                  tmp2 += tmp;
                } // q
                u[k].entry(i,j) += tmp2;
              }
            } // p
          }
        } // s

      } // j
    }, "hessvec_ktensor_kernel");
  }
}

}

template <typename ExecSpace>
void hess_vec(const SptensorT<ExecSpace>& X,
              const KtensorT<ExecSpace>& a,
              const KtensorT<ExecSpace>& v,
              const KtensorT<ExecSpace>& u,
              const AlgParams& algParams)
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::hess_vec");
#endif

  const ttb_indx nc = a.ncomponents();     // Number of components
  const ttb_indx nd = a.ndims();           // Number of dimensions

  assert(X.ndims() == nd);
  assert(v.ndims() == nd);
  assert(v.ncomponents() == nc);
  assert(u.ndims() == nd);
  assert(u.ncomponents() == nc);
  assert(v.isConsistent());
  assert(u.isConsistent());
  for (ttb_indx i=0; i<nd; ++i) {
    assert(a[i].nRows() == X.size(i));
    assert(v[i].nRows() == X.size(i));
    assert(u[i].nRows() == X.size(i));
  }

  using Kokkos::Experimental::ScatterDuplicated;
  using Kokkos::Experimental::ScatterNonDuplicated;
  using Kokkos::Experimental::ScatterAtomic;
  using Kokkos::Experimental::ScatterNonAtomic;
  typedef SpaceProperties<ExecSpace> space_prop;

  // Compute the first (tensor) term in the hess-vec
  Hess_Vec_Tensor_Method::type method = algParams.hess_vec_tensor_method;
  if (space_prop::is_gpu &&
      (method == Hess_Vec_Tensor_Method::Single ||
       method == Hess_Vec_Tensor_Method::Duplicated))
    Genten::error("Single and duplicated hess-vec tensor methods are invalid on Cuda and HIP!");

  if (method == Hess_Vec_Tensor_Method::Single) {
    Impl::HessVec_Kernel<ScatterNonDuplicated,ScatterNonAtomic,ExecSpace> kernel(X,a,v,u,algParams);
    Impl::run_row_simd_kernel(kernel, nc);
  }
  else if (method == Hess_Vec_Tensor_Method::Atomic) {
    Impl::HessVec_Kernel<ScatterNonDuplicated,ScatterAtomic,ExecSpace> kernel(X,a,v,u,algParams);
    Impl::run_row_simd_kernel(kernel, nc);
  }
  else if (method == Hess_Vec_Tensor_Method::Duplicated) {
    Impl::HessVec_Kernel<ScatterDuplicated,ScatterNonAtomic,ExecSpace> kernel(X,a,v,u,algParams);
    Impl::run_row_simd_kernel(kernel, nc);
  }
  else if (method == Hess_Vec_Tensor_Method::Perm) {
    if (!X.havePerm())
      Genten::error("Perm hess-vec tensor method selected, but permutation array not computed!");
    Impl::HessVec_PermKernel<ExecSpace> kernel(X,a,v,u,algParams);
    Impl::run_row_simd_kernel(kernel, nc);
  }
  else
    Genten::error(std::string("Invalid mttkrp-all-method for hess-vec:  ") +
                  MTTKRP_All_Method::names[algParams.mttkrp_all_method]);

  const ProcessorMap *pmap = u.getProcessorMap();
  if (pmap != nullptr) {
    Kokkos::fence();
    for (ttb_indx n=0; n<nd; ++n)
      pmap->subGridAllReduce(n, u[n].view().data(), u[n].view().span());
  }

  // Scale first term by -1
  for (unsigned n=0; n<nd; ++n)
    u[n].times(-1.0);

  // Add in the second (ktensor) term
  Impl::hess_vec_ktensor_term(a, v, u);
}

template <typename ExecSpace>
void hess_vec(const TensorT<ExecSpace>& X,
              const KtensorT<ExecSpace>& a,
              const KtensorT<ExecSpace>& v,
              const KtensorT<ExecSpace>& u,
              const AlgParams& algParams)
{
  #ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::hess_vec");
#endif

  const ttb_indx nc = a.ncomponents();     // Number of components
  const ttb_indx nd = a.ndims();           // Number of dimensions

  assert(X.ndims() == nd);
  assert(v.ndims() == nd);
  assert(v.ncomponents() == nc);
  assert(u.ndims() == nd);
  assert(u.ncomponents() == nc);
  assert(v.isConsistent());
  assert(u.isConsistent());
  for (ttb_indx i=0; i<nd; ++i) {
    assert(a[i].nRows() == X.size(i));
    assert(v[i].nRows() == X.size(i));
    assert(u[i].nRows() == X.size(i));
  }

  // Compute first (tensor) term
  Impl::HessVec_Dense_Kernel<ExecSpace> kernel(X,a,v,u,algParams);
  Impl::run_row_simd_kernel(kernel, nc);

  // Scale first term by -1
  for (unsigned n=0; n<nd; ++n)
    u[n].times(-1.0);

  // Add in the second (ktensor) term
  Impl::hess_vec_ktensor_term(a, v, u);

  // allReduce disabled because TensorT doesn't yet support MPI parallelism
  // if (u.getProcessorMap() != nullptr) {
  //   Kokkos::fence();
  //   for (ttb_indx n=0; n<nd; ++n)
  //     u.getProcessorMap()->subGridAllReduce(n, u[n].view().data(),
  //                                           u[n].view().span());
  // }
}

template <typename TensorType>
void gauss_newton_hess_vec(const TensorType& X,
                           const KtensorT<typename TensorType::exec_space>& a,
                           const KtensorT<typename TensorType::exec_space>& v,
                           const KtensorT<typename TensorType::exec_space>& u,
                           const AlgParams& algParams)
{
  typedef typename TensorType::exec_space ExecSpace;

  const ProcessorMap *pmap = u.getProcessorMap();
  const ttb_indx nc = a.ncomponents();     // Number of components
  const ttb_indx nd = a.ndims();           // Number of dimensions

  assert(X.ndims() == nd);
  assert(v.ndims() == nd);
  assert(v.ncomponents() == nc);
  assert(u.ndims() == nd);
  assert(u.ncomponents() == nc);
  assert(v.isConsistent());
  assert(u.isConsistent());
  for (ttb_indx i=0; i<nd; ++i) {
    assert(a[i].nRows() == X.size(i));
    assert(v[i].nRows() == X.size(i));
    assert(u[i].nRows() == X.size(i));
  }

  IndxArrayT<ExecSpace> nrow(nd, nc);
  FacMatArrayT<ExecSpace> Z(nd, nrow, nc) , W(nd, nrow, nc);
  for (unsigned n=0; n<nd; ++n) {
    Z[n].gramian(a[n], true); // full gram
    W[n].gemm(true, false, 1.0, a[n], v[n], 0.0);  // a[n]^T * v[n]
    if (pmap != nullptr)
      pmap->facMap(n)->allReduce(W[n].view().data(), W[n].view().span());
  }

  for (unsigned k=0; k<nd; ++k) {
    const ttb_indx I_k = X.size(k);
    Kokkos::RangePolicy<ExecSpace> policy(0,I_k);
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ttb_indx i)
    {
      for (unsigned j=0; j<nc; ++j) {
        for (unsigned s=0; s<nd; ++s) {
          const ttb_indx I_s = X.size(s);
          if (s == k) {
            for (unsigned p=0; p<nc; ++p) {
              ttb_real tmp = 1.0;
              for (unsigned n=0; n<nd; ++n) {
                if (n != k)
                  tmp *= Z[n].entry(p,j);
              } // n
              u[k].entry(i,j) += tmp * v[s].entry(i,p);
            } // p
          }
          else {
            for (unsigned p=0; p<nc; ++p) {
              ttb_real tmp = a[k].entry(i,p)*W[s].entry(j,p);
              for (unsigned n=0; n<nd; ++n) {
                if (n != k && n != s)
                  tmp *= Z[n].entry(p,j);
              }
              u[k].entry(i,j) += tmp;
            } // p
          }
        } // s

      } // j
    }, "hessvec_ktensor_kernel");
  }
}

}

#define INST_MACRO(SPACE)                                               \
  template                                                              \
  void hess_vec(const SptensorT<SPACE>& X,                              \
    const KtensorT<SPACE>& a,                                           \
    const KtensorT<SPACE>& v,                                           \
    const KtensorT<SPACE>& u,                                           \
    const AlgParams& algParams);                                        \
                                                                        \
  template                                                              \
  void hess_vec(const TensorT<SPACE>& X,                                \
    const KtensorT<SPACE>& a,                                           \
    const KtensorT<SPACE>& v,                                           \
    const KtensorT<SPACE>& u,                                           \
    const AlgParams& algParams);                                        \
                                                                        \
  template                                                              \
  void gauss_newton_hess_vec(const SptensorT<SPACE>& X,                 \
    const KtensorT<SPACE>& a,                                           \
    const KtensorT<SPACE>& v,                                           \
    const KtensorT<SPACE>& u,                                           \
    const AlgParams& algParams);                                        \
                                                                        \
  template                                                              \
  void gauss_newton_hess_vec(const TensorT<SPACE>& X,                   \
    const KtensorT<SPACE>& a,                                           \
    const KtensorT<SPACE>& v,                                           \
    const KtensorT<SPACE>& u,                                           \
    const AlgParams& algParams);
GENTEN_INST(INST_MACRO)