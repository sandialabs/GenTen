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
// ************************************************************************
//@HEADER


#pragma once
#include <assert.h>

#include "Genten_FacMatArray.h"
#include "Genten_IndxArray.h"
#include "Genten_RandomMT.h"
#include "Genten_Util.h"

namespace Genten
{

class Ktensor
{
public:

  // ----- CREATE & DESTROY -----
  // Empty constructor
  KOKKOS_INLINE_FUNCTION
  Ktensor() = default;

  // Constructor with number of components and dimensions, but
  // factor matrix sizes are still undetermined.
  Ktensor(ttb_indx nc, ttb_indx nd);

  // Constructor with number of components, dimensions and factor matrix sizes
  Ktensor(ttb_indx nc, ttb_indx nd, const Genten::IndxArray & sz);

  // Destructor
  KOKKOS_INLINE_FUNCTION
  ~Ktensor() = default;

  // Copy constructor
  KOKKOS_INLINE_FUNCTION
  Ktensor (const Ktensor & arg) = default;

  // Assignment operator
  KOKKOS_INLINE_FUNCTION
  Ktensor & operator= (const Ktensor & arg) = default;

  // ----- MODIFY & RESET -----

  // Set all entries to random values between 0 and 1.
  // Does not change the matrix array, so the Ktensor can become inconsistent.
  void setWeightsRand();

  // Set all weights equal to val.
  void setWeights(ttb_real val);

  // Set all weights to new values.  The length of newWeights must equal
  // that of the weights, as returned by ncomponents().
  void setWeights(const Genten::Array & newWeights);

  // Set all matrix entries equal to val
  void setMatrices(ttb_real val);

  // Set all entries to random values in [0,1).
  /*!
   *  Does not change the weights array, so the Ktensor can become inconsistent.
   *
   *  A new stream of Mersenne twister random numbers is generated, starting
   *  from an arbitrary seed value.  Use setMatricesScatter() for
   *  reproducibility.
   */
  void setMatricesRand();

  // Set all entries to reproducible random values.
  /*!
   *  Does not change the weights array, so the Ktensor can become inconsistent.
   *
   *  A new stream of Mersenne twister random numbers is generated, starting
   *  from an arbitrary seed value.  Use scatter() for reproducibility.
   *

   *  @param[in] bUseMatlabRNG  If true, then generate random samples
   *                            consistent with Matlab (costs twice as much
   *                            compared with no Matlab consistency).
   *  @param[in] cRMT           Mersenne Twister random number generator.
   *                            The seed should already be set.
   */
  void setMatricesScatter(const bool bUseMatlabRNG,
                          RandomMT & cRMT);

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
                         Genten::RandomMT & cRMT);

  // multiply (plump) a fraction (indices randomly chosen) of each FacMatrix by scale
  // If columnwise is true, each the selection process is applied columnwise.
  // If not columnwise, the selection is over the entire FacMatrix which may
  // yield some columns with no raisins.
  // For large column sizes and good RNGs, there may be no difference.
  // For small column sizes, inappropriate fractions may generate a warning.
  void scaleRandomElements(ttb_real fraction, ttb_real scale, bool columnwise);


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

  // Consistency check on sizes, i.e., the number of columns in each matrix
  // is equal to the length of lambda
  bool isConsistent() const;

  // Consistency check on sizes --- Same as above but also checks that the
  // number of rows in each matrix matches the specified size.
  bool isConsistent(const Genten::IndxArray & sz) const;

  bool hasNonFinite(ttb_indx &bad) const;

  // Return true if the ktensor is nonnegative: no negative entries and
  // no negative weights.  Optionally print the first error found.
  bool isNonnegative(bool bDisplayErrors) const;


  // ----- ELEMENT ACCESS -----

  // Return reference to weights vector
  KOKKOS_INLINE_FUNCTION
  Genten::Array & weights()
  {
    return(lambda);
  }

  // Return reference to weights vector
  KOKKOS_INLINE_FUNCTION
  const Genten::Array & weights() const
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
  const Genten::FacMatArray & factors() const
  {
    return data;
  }

  // Return a reference to the n-th factor matrix.
  // Factor matrices reference a component vector by column, and an element
  // within a component vector by row.  The number of columns equals the
  // number of components, and the number of rows equals the length of factors
  // in the n-th dimension.
  KOKKOS_INLINE_FUNCTION
  Genten::FacMatrix & operator[](ttb_indx n) const
  {
    assert((n >= 0) && (n < ndims()));
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
  bool isEqual(const Genten::Ktensor & b, ttb_real tol) const;

  // Return entry of constructed Ktensor.
  ttb_real entry(const Genten::IndxArray & subs) const;

  // Return entry of constructed Ktensor, substituting altLambda for lambda.
  ttb_real entry(const Genten::IndxArray & subs,
                 const Genten::Array & altLambda);

  // Distribute weights to i-th factor matrix (set lambda to vector of ones)
  void distribute(ttb_indx i);

  // Normalize i-th factor matrix using specified norm
  void normalize(Genten::NormType norm_type, ttb_indx i);

  // Normalize the factor matrices using the specified norm type
  void normalize(Genten::NormType norm_type);

  // Arrange the columns of the factor matrices by decreasing lambda value
  void arrange();

  // Arrange the columns of the factor matrices using a particular index permutation
  void arrange(Genten::IndxArray permutation_indices);

  // Return the Frobenius norm squared (sum of squares of each tensor element).
  ttb_real normFsq() const;


private:

  // Weights array
  Genten::Array lambda;

  // Factor matrix array.
  // See comments for access method operator[].
  Genten::FacMatArray data;

};

}
