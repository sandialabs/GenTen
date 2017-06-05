//@HEADER
// ************************************************************************
//     Genten Tensor Toolbox
//     Software package for tensor math by Sandia National Laboratories
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


/*!
  @file Genten_FacTestSetGenerator.h
  @brief Class declaration for a test set generator.
*/


#ifndef GENTEN_FACTESTSETGENERATOR_H
#define GENTEN_FACTESTSETGENERATOR_H

#include "Genten_IndxArray.h"
#include "Genten_Ktensor.h"
#include "Genten_RandomMT.h"
#include "Genten_Sptensor.h"

namespace Genten
{


//----------------------------------------------------------------------
//! Generates tensor test sets compatible with a factorization.
/*!
 *  Test sets are normally a pair of tensors: one providing data and
 *  the other providing an expected solution.  The solution is based
 *  on a factorization (CP or Tucker).
 *
 *  It is desirable to match Matlab test sets.  For this reason, a common
 *  random number generator can be used.
 */
//----------------------------------------------------------------------
  class FacTestSetGenerator
  {
  public:

    //! Constructor.
    FacTestSetGenerator (void);

    //! Destructor.
    ~FacTestSetGenerator (void);

    //! Generate sparse tensor data from a random Ktensor factorization.
    /*!
     *  Expected factors are created by populating elements with [0,1]
     *  random variables.
     *  Nonzero elements of the data tensor are chosen by sampling from
     *  stochastic vectors, one for each factor vector.  Values in the sparse
     *  tensor output equal the expected count; hence, the underlying random
     *  process is Poisson-based.
     *
     *  <pre>
     *  The same results can be generated in Matlab for comparison:
     *    rstream = RandStream('mt19937ar', 'Seed',nSeed);
     *    RandStream.setGlobalStream(rstream);
     *    X = create_problem('Size', cDims, ...
     *                       'Num_Factors', nNumComps, ...
     *                       'Factor_Generator', @rand, ...
     *                       'Lambda_Generator', @rand, ...
     *                       'Sparse_Generation', nMaxNnz);
     *    X.Data corresponds to cDataTensor
     *    X.Soln corresponds to cExpectedFactors
     *  </pre>
     *
     *  @param[in] cDims              Array providing shape of tensors.
     *  @param[in] nNumComps          Number of components in Ktensor factors.
     *  @param[in] nMaxNnz            Number of nonzeroes generated, but
     *                                cDataTensor may have fewer if there are
     *                                duplicates.
     *  @param[in] cRNG               Random number generator.
     *  @param[out] cDataTensor       Sparse data tensor.  Number of nonzeroes
     *                                is <= nMaxNnz.  Value of a nonzero is
     *                                the count of entries generated.
     *  @param[out] cExpectedFactors  Factorization from which cDataTensor
     *                                was derived.
     *  @return  True if successful.
     */
    bool  genSpFromRndKtensor (const IndxArray &  cDims,
                               const ttb_indx     nNumComps,
                               const ttb_indx     nMaxNnz,
                               RandomMT  &  cRNG,
                               Sptensor  &  cDataTensor,
                               Ktensor   &  cExpectedFactors) const;

    //! Generate sparse tensor data from a boosted random Ktensor factorization.
    /*!
     *  Expected factors are created by populating elements with [0,1]
     *  random variables using a boosting formula.  Elements are initially
     *  either dSmallValue, or a random value in the [1,dMaxValue].  The
     *  fraction of boosted values is given by dFracBoosted.  These initial
     *  factors are given random weights and then normalized so elements
     *  are in [0,1].
     *  Nonzero elements of the data tensor are chosen by sampling from
     *  stochastic vectors, one for each factor vector.  Values in the sparse
     *  tensor output equal the expected count; hence, the underlying random
     *  process is Poisson-based.
     *
     *  <pre>
     *  The same results can be generated in Matlab for comparison:
     *    rstream = RandStream('mt19937ar', 'Seed',nSeed);
     *    RandStream.setGlobalStream(rstream);
     *    X = create_problem('Size', cDims, ...
     *                       'Num_Factors', nNumComps, ...
     *                       'Factor_Generator', ...
     *                         boostSp(m,n,dFracBoosted,dMaxValue,dSmallValue), ...
     *                       'Lambda_Generator', @rand, ...
     *                       'Sparse_Generation', nMaxNnz);
     *    X.Data corresponds to cDataTensor
     *    X.Soln corresponds to cExpectedFactors
     *
     *    function A = boostSp (N,R,fracBoosted,maxValue,smallValue)
     *      A = zeros(N,R);
     *      n = round(fracBoosted * N);
     *      for r = 1:R
     *        v = smallValue * ones(N,1);
     *        v(1:n) = 1 + (maxValue - 1) * rand(n,1);
     *        p = randperm(N);
     *        A(:,r) = v(p(1:N));
     *      end
     *  </pre>
     *
     *  @param[in] cDims              Array providing shape of tensors.
     *  @param[in] nNumComps          Number of components in Ktensor factors.
     *  @param[in] nMaxNnz            Number of nonzeroes generated, but
     *                                cDataTensor may have fewer if there are
     *                                duplicates.
     *  @param[in] dFracBoosted       Fraction of factor elements boosted.
     *  @param[in] dMaxValue          Maximum value for a boosted element.
     *  @param[in] dSmallValue        Value for all non-boosted elements.
     *  @param[in] cRNG               Random number generator.
     *  @param[out] cDataTensor       Sparse data tensor.  Number of nonzeroes
     *                                is <= nMaxNnz.  Value of a nonzero is
     *                                the count of entries generated.
     *  @param[out] cExpectedFactors  Factorization from which cDataTensor
     *                                was derived.
     *  @return  True if successful.
     */
    bool  genSpFromBoostedRndKtensor (const IndxArray &  cDims,
                                      const ttb_indx     nNumComps,
                                      const ttb_indx     nMaxNnz,
                                      const ttb_real     dFracBoosted,
                                      const ttb_real     dMaxValue,
                                      const ttb_real     dSmallValue,
                                      RandomMT  &  cRNG,
                                      Sptensor  &  cDataTensor,
                                      Ktensor   &  cExpectedFactors) const;


  private:

    //! By design, there is no copy constructor.
    FacTestSetGenerator (const FacTestSetGenerator &);
    //! By design, there is no assignment operator.
    FacTestSetGenerator & operator= (const FacTestSetGenerator &);

  };

}          //-- namespace Genten

#endif     //-- GENTEN_FACTESTSETGENERATOR_H
