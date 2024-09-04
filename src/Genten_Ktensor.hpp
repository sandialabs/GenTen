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
#include <cassert>
#include <any>

#include "Genten_FacMatArray.hpp"
#include "Genten_IndxArray.hpp"
#include "Genten_RandomMT.hpp"
#include "Genten_Util.hpp"
#include "Genten_TinyVec.hpp"
#include "Genten_SimdKernel.hpp"
#include "Genten_Pmap.hpp"

namespace Genten
{

template <typename ExecSpace> class KtensorT;
typedef KtensorT<DefaultHostExecutionSpace> Ktensor;

template <typename ExecSpace>
class KtensorImpl
{
public:

  typedef ExecSpace exec_space;
  typedef typename ArrayT<ExecSpace>::host_mirror_space host_mirror_space;
  typedef KtensorImpl<host_mirror_space> HostMirror;

  // ----- CREATE & DESTROY -----
  // Empty constructor
  KOKKOS_DEFAULTED_FUNCTION
  KtensorImpl() = default;

  // Constructor with number of components and dimensions, but
  // factor matrix sizes are still undetermined.
  KtensorImpl(ttb_indx nc, ttb_indx nd, const ProcessorMap* pmap_ = nullptr);

  // Constructor with number of components, dimensions and factor matrix sizes
  KtensorImpl(ttb_indx nc, ttb_indx nd, const IndxArrayT<ExecSpace> & sz,
             const ProcessorMap* pmap_ = nullptr);

  // Create Ktensor from supplied weights and values
  KOKKOS_INLINE_FUNCTION
  KtensorImpl(const ArrayT<ExecSpace>& w, const FacMatArrayT<ExecSpace>& vals,
             const ProcessorMap* pmap_ = nullptr) :
    lambda(w), data(vals), pmap(pmap_) {}

  // Destructor
  KOKKOS_DEFAULTED_FUNCTION
  ~KtensorImpl() = default;

  // Copy constructor
  KOKKOS_DEFAULTED_FUNCTION
  KtensorImpl (const KtensorImpl & arg) = default;

  // Assignment operator
  KOKKOS_DEFAULTED_FUNCTION
  KtensorImpl & operator= (const KtensorImpl & arg) = default;

  // ----- MODIFY & RESET -----

  // Set all entries to random values between 0 and 1.
  // Does not change the matrix array, so the Ktensor can become inconsistent.
  void setWeightsRand() const;

  // Set all weights equal to val.
  void setWeights(ttb_real val) const;

  // Set all weights to new values.  The length of newWeights must equal
  // that of the weights, as returned by ncomponents().
  void setWeights(const ArrayT<ExecSpace> & newWeights) const;

  // Set all matrix entries equal to val
  void setMatrices(ttb_real val) const;

  // Set all entries to random values in [0,1).
  /*!
   *  Does not change the weights array, so the Ktensor can become inconsistent.
   *
   *  A new stream of Mersenne twister random numbers is generated, starting
   *  from an arbitrary seed value.  Use setMatricesScatter() for
   *  reproducibility.
   */
  void setMatricesRand() const;

  // Set all entries to reproducible random values.
  /*!
   *  Does not change the weights array, so the Ktensor can become inconsistent.
   *
   *  A new stream of Mersenne twister random numbers is generated, starting
   *  from an arbitrary seed value.  Use scatter() for reproducibility.
   *
   *
   *  @param[in] bUseMatlabRNG   If true, then generate random samples
   *                             consistent with Matlab (costs twice as much
   *                             compared with no Matlab consistency).
   *  @param[in] bUseParallelRNG If true, then generate random samples in
   *                             parallel (resulting random number sequence
   *                             will depend on number of threads and
   *                             aarchitecture).
   *  @param[in] cRMT            Mersenne Twister random number generator.
   *                             The seed should already be set.
   */
  void setMatricesScatter(const bool bUseMatlabRNG,
                          const bool bUseParallelRNG,
                          RandomMT & cRMT) const;

  //! Fill the Ktensor with uniform random values, normalized to be stochastic.
  /*!
   *  Fill each factor matrix with random variables, uniform from [0,1),
   *  scale each vector so it sums to 1 (adjusting the weights), apply
   *  random factors to the weights, and finally normalize weights.
   *  The result has stochastic columns in all factors and weights that
   *  sum to 1.
   *
   *  Random values can be chosen consistent with those used by Matlab Genten
   *  function create_problem.
   *  Mersenne twister is used to generate all [0,1) random samples.
   *
   *  @param[in] bUseMatlabRNG  If true, then generate random samples
   *                            consistent with Matlab (costs twice as much
   *                            compared with no Matlab consistency).
   *  @param[in] cRMT           Mersenne Twister random number generator.
   *                            The seed should already be set.
   */
  void setRandomUniform (const bool bUseMatlabRNG,
                         RandomMT & cRMT) const;

  // multiply (plump) a fraction (indices randomly chosen) of each FacMatrix by scale
  // If columnwise is true, each the selection process is applied columnwise.
  // If not columnwise, the selection is over the entire FacMatrix which may
  // yield some columns with no raisins.
  // For large column sizes and good RNGs, there may be no difference.
  // For small column sizes, inappropriate fractions may generate a warning.
  void scaleRandomElements(ttb_real fraction, ttb_real scale, bool columnwise) const;

  void setProcessorMap(const ProcessorMap* pmap_);
  const ProcessorMap* getProcessorMap() const { return pmap; }


  // ----- PROPERTIES -----

  // Return number of components
  KOKKOS_INLINE_FUNCTION
  ttb_indx ncomponents() const
  {
    return(lambda.size());
  }

  // Return number of dimensions of Ktensor
  KOKKOS_INLINE_FUNCTION
  ttb_indx ndims() const
  {
    return(data.size());
  }

  // Return if ktensor is empty
  KOKKOS_INLINE_FUNCTION
  bool isEmpty() const {
    return ncomponents() == 0 || ndims() == 0;
  }

  // Consistency check on sizes, i.e., the number of columns in each matrix
  // is equal to the length of lambda
  bool isConsistent() const;

  // Consistency check on sizes --- Same as above but also checks that the
  // number of rows in each matrix matches the specified size.
  bool isConsistent(const IndxArrayT<ExecSpace> & sz) const;

  // Return true if the ktensor is nonnegative: no negative entries and
  // no negative weights.  Optionally print the first error found.
  bool isNonnegative(bool bDisplayErrors) const;


  // ----- ELEMENT ACCESS -----

  // Return reference to weights vector
  KOKKOS_INLINE_FUNCTION
  ArrayT<ExecSpace> & weights()
  {
    return(lambda);
  }

  // Return reference to weights vector
  KOKKOS_INLINE_FUNCTION
  const ArrayT<ExecSpace> & weights() const
  {
    return(lambda);
  }

  // Return reference to element i of weights vector
  KOKKOS_INLINE_FUNCTION
  ttb_real & weights(ttb_indx i) const
  {
    assert(i < lambda.size() );
    return(lambda[i]);
  }

  // Return reference to factor matrix array
  KOKKOS_INLINE_FUNCTION
  const FacMatArrayT<ExecSpace> & factors() const
  {
    return data;
  }

  KOKKOS_INLINE_FUNCTION
  FacMatArrayT<ExecSpace> & factors()
  {
    return data;
  }

  // Set a factor matrix
  void set_factor(const ttb_indx i, const FacMatrixT<ExecSpace>& src) const
  {
    data.set_factor(i, src);
  }

  // Return a reference to the n-th factor matrix.
  // Factor matrices reference a component vector by column, and an element
  // within a component vector by row.  The number of columns equals the
  // number of components, and the number of rows equals the length of factors
  // in the n-th dimension.
  KOKKOS_INLINE_FUNCTION
  const FacMatrixT<ExecSpace>& operator[](ttb_indx n) const
  {
    assert(n < ndims());
    return data[n];
  }


  // ----- FUNCTIONS -----

  // Return true if this Ktensor is equal to b within a specified tolerance.
  /* Being equal means that the two Ktensors are the same size and
   * all weights and factor matrix elements satisfy

                fabs(a(i,j) - b(i,j))
           ---------------------------------   < TOL .
           max(1, fabs(a(i,j)), fabs(b(i,j))
  */
  bool isEqual(const KtensorImpl & b, ttb_real tol) const;

  // Return entry of constructed Ktensor.
  ttb_real entry(const IndxArrayT<ExecSpace> & subs) const;

  // Distribute weights uniformly across factor matrices (set lambda to vector of ones)
  void distribute() const;

  // Distribute weights to i-th factor matrix (set lambda to vector of ones)
  void distribute(ttb_indx i) const;

  // Normalize i-th factor matrix using specified norm
  void normalize(NormType norm_type, ttb_indx i) const;

  // Normalize the factor matrices using the specified norm type
  void normalize(NormType norm_type) const;

  // Arrange the columns of the factor matrices by decreasing lambda value
  void arrange() const;

  // Arrange the columns of the factor matrices using a particular index permutation
  void arrange(const IndxArray& permutation_indices) const;

  // Return the Frobenius norm squared (sum of squares of each tensor element).
  ttb_real normFsq() const;

  // Return the Frobenius norm squared (sum of squares of each tensor element)
  // using specified weights array w
  ttb_real normFsq(const ArrayT<ExecSpace> & w) const;

  // Compute the Khatri-Rao product of the factors given by modes
  FacMatrixT<ExecSpace> khatrirao(const std::vector<ttb_indx>& modes) const;

private:

  // Weights array
  ArrayT<ExecSpace> lambda;

  // Factor matrix array.
  // See comments for access method operator[].
  FacMatArrayT<ExecSpace> data;

  const ProcessorMap* pmap;

};

template <typename ExecSpace>
class KtensorT : public KtensorImpl<ExecSpace>
{
public:

  using impl_type = KtensorImpl<ExecSpace>;
  using exec_space = typename impl_type::exec_space;
  using host_mirror_space = typename impl_type::host_mirror_space;
  using HostMirror = KtensorT<host_mirror_space>;

  using impl_type::impl_type;

  KtensorT() {}
  KtensorT(KtensorT&&) = default;
  KtensorT(const KtensorT&) = default;
  KtensorT& operator=(KtensorT&&) = default;
  KtensorT& operator=(const KtensorT&) = default;
  ~KtensorT() {}

  impl_type& impl() { return *this; }
  const impl_type& impl() const { return *this; }

  // For passing extra data, like numpy arrays, through
  template <typename T>
  void set_extra_data(const T& a) {
    extra_data = std::make_any<T>(a);
  }
  bool has_extra_data() const {
    return extra_data.has_value();
  }
  template <typename T>
  bool has_extra_data_type() const {
    return extra_data.has_value() && (std::any_cast<T>(&extra_data) != nullptr);
  }
  template <typename T>
  T get_extra_data() const {
    gt_assert(extra_data.has_value());
    return std::any_cast<T>(extra_data);
  }
  template <typename E>
  void copy_extra_data(const KtensorT<E>& x) {
    // only copy extra data if this and x point to the same data
    if (this->factors().is_same(x.factors()))
      extra_data = x.extra_data;
  }

protected:

  std::any extra_data;
  template <typename E> friend class KtensorT;
};

template <typename ExecSpace>
typename KtensorT<ExecSpace>::HostMirror
create_mirror_view(const KtensorT<ExecSpace>& a)
{
  typedef typename KtensorT<ExecSpace>::HostMirror HostMirror;
  HostMirror hm( create_mirror_view(a.weights()),
                 create_mirror_view(a.factors()),
                 a.getProcessorMap() );
  hm.copy_extra_data(a);
  return hm;
}

template <typename Space, typename ExecSpace>
KtensorT<Space>
create_mirror_view(const Space& s, const KtensorT<ExecSpace>& a)
{
  KtensorT<Space> v( create_mirror_view(s, a.weights()),
                     create_mirror_view(s, a.factors()),
                     a.getProcessorMap() );
  v.copy_extra_data(a);
  return v;
}

template <typename E1, typename E2>
void deep_copy(const KtensorT<E1>& dst, const KtensorT<E2>& src)
{
  deep_copy( dst.weights(), src.weights() );
  deep_copy( dst.factors(), src.factors() );
}

template <typename ExecSpace> class SptensorImpl;

// Compute Ktensor value using Sptensor subscripts
// Len and WarpOrWavefrontSize are for nested parallelism using TinyVec
template <typename ExecSpace, unsigned Len, unsigned WarpOrWavefrontSize>
KOKKOS_INLINE_FUNCTION
ttb_real compute_Ktensor_value(const typename Kokkos::TeamPolicy<ExecSpace>::member_type& team,
                               const KtensorImpl<ExecSpace>& M,
                               const SptensorImpl<ExecSpace>& X,
                               const ttb_indx i) {
  typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, Len, Len, WarpOrWavefrontSize> TVM1;

  /*const*/ unsigned nd = M.ndims();
  /*const*/ unsigned nc = M.ncomponents();

  auto m_val = TVM1::make(team, Len, 0.0);

  auto row_func = [&](auto j, auto nj, auto Nj) {
    typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, Len, Nj(), WarpOrWavefrontSize> TVM2;
    auto tmp = TVM2::make(team, nj, 0.0);

    tmp.load(&(M.weights(j)));
    for (unsigned m=0; m<nd; ++m) {
      tmp *= &(M[m].entry(X.subscript(i,m),j));
    }
    m_val += tmp;
  };

  for (unsigned j=0; j<nc; j+=Len) {
    if (j+Len < nc) {
      const unsigned nj = Len;
      row_func(j, nj, std::integral_constant<unsigned,Len>());
    }
    else {
      const unsigned nj = nc-j;
      row_func(j, nj, std::integral_constant<unsigned,0>());
    }
  }

  return m_val.sum();
}

// Compute Ktensor value using Sptensor subscripts
// Len and WarpOrWavefrontSize are for nested parallelism using TinyVec
template <typename ExecSpace, unsigned Len, unsigned WarpOrWavefrontSize,
          typename IndexArray>
KOKKOS_INLINE_FUNCTION
ttb_real compute_Ktensor_value(const typename Kokkos::TeamPolicy<ExecSpace>::member_type& team,
                               const KtensorImpl<ExecSpace>& M,
                               const IndexArray& ind) {
  typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, Len, Len, WarpOrWavefrontSize> TVM1;

  /*const*/ unsigned nd = M.ndims();
  /*const*/ unsigned nc = M.ncomponents();

  auto m_val = TVM1::make(team, Len, 0.0);

  auto row_func = [&](auto j, auto nj, auto Nj) {
    typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, Len, Nj(), WarpOrWavefrontSize> TVM2;
    auto tmp = TVM2::make(team, nj, 0.0);
    tmp.load(&(M.weights(j)));
    for (unsigned m=0; m<nd; ++m) {
      tmp *= &(M[m].entry(ind[m],j));
    }
    m_val += tmp;
  };

  for (unsigned j=0; j<nc; j+=Len) {
    if (j+Len < nc) {
      const unsigned nj = Len;
      row_func(j, nj, std::integral_constant<unsigned,Len>());
    }
    else {
      const unsigned nj = nc-j;
      row_func(j, nj, std::integral_constant<unsigned,0>());
    }
  }

  return m_val.sum();
}

// Compute Ktensor value using supplied subscripts
// Assumes flat parallelism
template <typename ExecSpace, typename IndexArray>
KOKKOS_INLINE_FUNCTION
ttb_real compute_Ktensor_value(const KtensorImpl<ExecSpace>& M,
                               const IndexArray& ind) {
  const unsigned nd = M.ndims();
  const unsigned nc = M.ncomponents();

  ttb_real m_val = 0.0;
  for (unsigned j=0; j<nc; ++j) {
    ttb_real tmp = M.weights(j);
    for (unsigned m=0; m<nd; ++m) {
      tmp *= M[m].entry(ind[m],j);
    }
    m_val += tmp;
  }

  return m_val;
}

template <typename ExecSpace>
KtensorT<ExecSpace> clone(const KtensorT<ExecSpace>& u)
{
  const ttb_indx nd = u.ndims();
  const ttb_indx nc = u.ncomponents();
  KtensorT<ExecSpace> v(nc,nd,u.getProcessorMap());
  for (ttb_indx i=0; i<nd; ++i) {
    FacMatrixT<ExecSpace> mat(u[i].nRows(), nc);
    v.set_factor(i,mat);
  }
  return v;
}

}
