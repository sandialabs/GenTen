//@HEADER
// ************************************************************************
//     Genten Tensor Toolbox
//     Software package for tensor math by Sandia National Laboratories
//
// Sandia National Laboratories is a multimission laboratory managed
// and operated by National Technology and Engineering Solutions of Sandia,
// LLC, a wholly owned subsidiary of Honeywell International, Inc., for the
// U.S. Department of Energy’s National Nuclear Security Administration under
// contract DE-NA0003525.
//
// Copyright 20XX, Sandia Corporation.
// ************************************************************************
//@HEADER


#include <assert.h>
#include <algorithm>
#include <iostream>

#include "Genten_Ktensor.h"
#include "Genten_RandomMT.h"
#include "Genten_IndxArray.h"

Genten::Ktensor::
Ktensor(ttb_indx nc, ttb_indx nd):
  lambda(nc), data(nd)
{
  setWeights(1.0);
}

Genten::Ktensor::
Ktensor(ttb_indx nc, ttb_indx nd, const Genten::IndxArray & sz):
  lambda(nc), data(nd,sz,nc)
{
  setWeights(1.0);
}

void Genten::Ktensor::
setWeightsRand()
{
  lambda.rand();
}

void Genten::Ktensor::
setWeights(ttb_real val)
{
  lambda = val;
}

void Genten::Ktensor::
setWeights(const Genten::Array &  newWeights)
{
  assert(newWeights.size() == lambda.size());
  for (ttb_indx  i = 0; i < lambda.size(); i++)
    lambda[i] = newWeights[i];
  return;
}



void Genten::Ktensor::
setMatricesRand()
{
  ttb_indx nd = data.size();
  for (ttb_indx n = 0; n < nd; n ++)
  {
    data[n].rand();
  }
}

void Genten::Ktensor::
setMatricesScatter(const bool bUseMatlabRNG,
                   Genten::RandomMT &   cRMT)
{
  ttb_indx nd = data.size();
  for (ttb_indx n = 0; n < nd; n ++)
  {
    data[n].scatter (bUseMatlabRNG, cRMT);
  }
}


void Genten::Ktensor::
setRandomUniform (const bool bUseMatlabRNG,
                  Genten::RandomMT & cRMT)
{
  // Set factor matrices to random values, then normalize each component
  // vector so that it sums to one.
  ttb_indx nComps = lambda.size();
  ttb_indx nd = data.size();
  Array  cTotals(nComps);
  setWeights (1.0);
  for(ttb_indx  n = 0; n < nd; n++)
  {
    cTotals = 0.0;
    for (ttb_indx  c = 0; c < nComps; c++)
    {
      ttb_indx nRows = data[n].nRows();
      for (ttb_indx  i = 0; i < nRows; i++)
      {
        ttb_real  dNextRan;
        if (bUseMatlabRNG)
          dNextRan = cRMT.genMatlabMT();
        else
          dNextRan = cRMT.genrnd_double();

        data[n].entry(i,c) = dNextRan;
        cTotals[c] += dNextRan;
      }
    }
    data[n].colScale (cTotals, true);
    for (ttb_indx  c = 0; c < nComps; c++)
      weights(c) *= cTotals[c];
  }

  // Adjust weights by a random factor.
  // Random values for weights are generated after all factor elements
  // to match Matlab Genten function create_problem.
  for (ttb_indx  c = 0; c < nComps; c++)
  {
    ttb_real  dNextRan;
    if (bUseMatlabRNG)
      dNextRan = cRMT.genMatlabMT();
    else
      dNextRan = cRMT.genrnd_double();

    weights(c) *= dNextRan;
  }

  // Normalize the weights so they sum to one.
  ttb_real  dTotal = 0.0;
  for (ttb_indx  c = 0; c < nComps; c++)
    dTotal += weights(c);
  for (ttb_indx  c = 0; c < nComps; c++)
    weights(c) *= (1.0 / dTotal);

  return;
}


// Only called by Ben Allan's parallel test code.
#if !defined(_WIN32)
void Genten::Ktensor::
scaleRandomElements(ttb_real fraction, ttb_real scale, bool columnwise)
{
  for (ttb_indx i =0; i< data.size(); i++) {
    data[i].scaleRandomElements(fraction, scale, columnwise);
  }
}
#endif

void Genten::Ktensor::
setMatrices(ttb_real val)
{
  data = val;
}

bool Genten::Ktensor::
isConsistent() const
{
  ttb_indx nc = lambda.size();
  for (ttb_indx n = 0; n < data.size(); n ++)
  {
    if (data[n].nCols() != nc)
    {
      return false;
    }
  }
  return true;
}

bool Genten::Ktensor::
isConsistent(const Genten::IndxArray & sz) const
{
  if (data.size() != sz.size())
  {
    return false;
  }

  ttb_indx nc = lambda.size();
  for (ttb_indx n = 0; n < data.size(); n ++)
  {
    if ((data[n].nCols() != nc) || (data[n].nRows() != sz[n]))
    {
      return false;
    }
  }
  return true;
}

bool Genten::Ktensor::
hasNonFinite(ttb_indx &bad) const
{
  bad = 0;
  if (lambda.hasNonFinite(bad)) {
    std::cout << "Genten::Ktensor::hasNonFinite lambda.data["<<bad<<"] nonfinite " << std::endl;
    return true;
  }
  for (ttb_indx i = 0; i < data.size(); i++) {
    if (data[i].hasNonFinite(bad)) {
      std::cout << "Genten::Ktensor::hasNonFinite data["<<i<<"] nonfinite element " << bad << std::endl;
      return true;
    }
  }

  return false;
}


bool Genten::Ktensor::
isNonnegative(bool bDisplayErrors) const
{
  for (ttb_indx  n = 0; n < ndims(); n++)
  {
    for (ttb_indx  i = 0; i < factors()[n].nRows(); i++)
    {
      for (ttb_indx  j = 0; j < ncomponents(); j++)
      {
        if (factors()[n].entry(i,j) < 0.0)
        {
          if (bDisplayErrors)
          {
            std::cout << "Ktensor::isNonnegative()"
                      << " - element (" << i << "," << j << ")"
                      << " of mode " << n << " is negative"
                      << std::endl;
          }
          return( false );
        }
      }
    }
  }
  for (ttb_indx  r = 0; r < ncomponents(); r++)
  {
    if (weights(r) < 0.0)
    {
      if (bDisplayErrors)
      {
        std::cout << "Ktensor::isNonnegative()"
                  << " - weight " << r << " is negative" << std::endl;
      }
      return( false );
    }
  }

  return( true );
}

bool Genten::Ktensor::
isEqual(const Genten::Ktensor & b, ttb_real tol) const
{
  // Check for equal sizes.
  if ((this->ndims() != b.ndims()) || (this->ncomponents() != b.ncomponents()))
  {
    return( false );
  }

  // Check for equal weights (within tolerance).
  if (this->weights().isEqual (b.weights(), tol) == false)
  {
    return( false );
  }

  // Check for equal factor matrices (within tolerance).
  for (ttb_indx  i = 0; i < ndims(); i++)
  {
    if (this->data[i].isEqual (b[i], tol) == false)
    {
      return( false );
    }
  }
  return( true );
}

ttb_real Genten::Ktensor::
entry(const Genten::IndxArray & subs) const
{
  ttb_indx nd = this->ndims();
  assert(subs.size() == nd);

  // This vector product is fundamental to many big computations; hence,
  // stride should be one.  Since FacMatrix stores by row, the factor vectors
  // are columns so that rowTimes() is across a row.

  // Copy lambda array to temp array.
  Genten::Array x(lambda.size());
  x.deep_copy(lambda);

  // Compute a vector of elementwise products of corresponding rows
  // of factor matrices.
  for (ttb_indx i = 0; i < nd; i ++)
  {
    // Update temp array with elementwise product.
    // If a subscript is out of bounds, it will be caught by rowTimes().
    data[i].rowTimes(x, subs[i]);
  }

  // Return sum of elementwise products stored in temp array.
  return(x.sum());
}

ttb_real Genten::Ktensor::
entry(const Genten::IndxArray & subs,
      const Genten::Array & altLambda)
{
  ttb_indx nd = this->ndims();
  assert(subs.size() == nd);
  assert(altLambda.size() == lambda.size());

  // This vector product is fundamental to many big computations; hence,
  // stride across lambda should be one.  Since FacMatrix stores by row,
  // the factor vectors are columns so that rowTimes() is across a row.

  // Copy lambda array to temp array.
  Genten::Array lambdaForEntry(lambda.size());
  for (ttb_indx  i = 0; i < lambdaForEntry.size(); i++)
    lambdaForEntry[i] = altLambda[i];

  // Compute a vector of elementwise products of corresponding rows
  // of factor matrices.
  for (ttb_indx i = 0; i < nd; i ++)
  {
    // Update temp array with elementwise product.
    // If a subscript is out of bounds, it will be caught by rowTimes().
    data[i].rowTimes (lambdaForEntry, subs[i]);
  }

  // Return sum of elementwise products stored in temp array.
  return( lambdaForEntry.sum() );
}

void Genten::Ktensor::
distribute(ttb_indx i)
{
  data[i].colScale(lambda,false);
  lambda = 1.0;
}

void Genten::Ktensor::
normalize(Genten::NormType norm_type, ttb_indx i)
{

#ifndef _GENTEN_CK_FINITE
#define CKFINITE 0 // set to 1 to turn on inf/nan checking.
#else
#define CKFINITE 1
#endif
  Genten::Array norms(lambda.size());
#if CKFINITE
  ttb_indx bad= 0;
  if (norms.hasNonFinite(bad)) {
    std::cout << " Genten::Ktensor::normalize bad norms element "<< bad << " at line " << __LINE__ << std::endl;
  }
  if (data[i].hasNonFinite(bad)) {
    std::cout << " Genten::Ktensor::normalize bad data["<<i<<"] element "<< bad << " at line " << __LINE__ << std::endl;
  }
#endif

  data[i].colNorms(norm_type, norms, 0.0);
  for (ttb_indx k = 0; k < norms.size(); k++)
  {
    if (norms[k] == 0)
    {
      norms[k] = 1;
    }
  }

#if CKFINITE
  if (data[i].hasNonFinite(bad)) {
    std::cout << " Genten::Ktensor::normalize bad data["<<i<<"] element "<< bad << " at line " << __LINE__ << std::endl;
  }
  if (norms.hasNonFinite(bad)) {
    std::cout << " Genten::Ktensor::normalize bad norms element "<< bad << " at line " << __LINE__ << std::endl;
  }
#endif

  data[i].colScale(norms, true);

#if CKFINITE
  if (data[i].hasNonFinite(bad)) {
    std::cout << " Genten::Ktensor::normalize bad data["<<i<<"] element "<< bad << " at line " << __LINE__ << std::endl;
  }
  if (norms.hasNonFinite(bad)) {
    std::cout << " Genten::Ktensor::normalize bad norms element "<< bad << " at line " << __LINE__ << std::endl;
  }
#endif

  lambda.times(norms);

#if CKFINITE
  if (lambda.hasNonFinite(bad)) {
    std::cout << " Genten::Ktensor::normalize bad lambda element "<< bad << " at line " << __LINE__ << std::endl;
  }
#endif
}


void Genten::Ktensor::
normalize(Genten::NormType norm_type)
{
// could be much better vectorized instead of walking memory data.size times.
  for (ttb_indx n = 0; n < data.size(); n ++)
  {
    this->normalize(norm_type, n);
  }
}


struct greater_than
{
  template<class T>
  bool operator()(T const &a, T const &b) const { return a.first > b.first; }
};

void Genten::Ktensor::
arrange()
{
  // sort lambda by value and keep track of sort index
  std::vector<std::pair<ttb_real,ttb_indx> > lambda_pair;
  for (ttb_indx i = 0 ; i != lambda.size() ; i++) {
    lambda_pair.push_back(std::make_pair(lambda[i], i));
  }
  sort(lambda_pair.begin(),lambda_pair.end(),greater_than());

  // create permuted indices
  Genten::IndxArray p(lambda.size());
  for (size_t i = 0 ; i != lambda.size() ; i++)
    p[i] = lambda_pair[i].second;

  // arrange the columns of the factor matrices using the permutation
  this->arrange(p);
}

void Genten::Ktensor::
arrange(Genten::IndxArray permutation_indices)
{
  // permute factor matrices
  for (ttb_indx n = 0; n < data.size(); n ++)
    data[n].permute(permutation_indices);

  // permute lambda values
  Genten::Array new_lambda(lambda.size());
  for (ttb_indx i = 0; i < lambda.size(); i ++)
    new_lambda[i] = lambda[permutation_indices[i]];
  for (ttb_indx i = 0; i < lambda.size(); i ++)
    lambda[i] = new_lambda[i];

}


ttb_real Genten::Ktensor::
normFsq() const
{
  ttb_real  dResult = 0.0;

  // This technique computes an RxR matrix of dot products between all factor
  // column vectors of each mode, then forms the Hadamard product of these
  // matrices.  The last step is the scalar \lambda' H \lambda.
  Genten::FacMatrix  cH(ncomponents(), ncomponents());
  cH = 1;
  Genten::FacMatrix  cG(ncomponents(),ncomponents());
  for (ttb_indx  n = 0; n < ndims(); n++)
  {
    cG.gramian(data[n]);
    cH.times(cG);
  }
  dResult = 0.0;
  for (ttb_indx  r = 0; r < ncomponents(); r++)
  {
    dResult += weights(r) * weights(r) * cH.entry(r,r);
    for (ttb_indx  q = r+1; q < ncomponents(); q++)
    {
      dResult += 2.0 * weights(r) * weights(q) * cH.entry(r,q);
    }
  }

  return( dResult );
}
