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


#include <algorithm>
#include <map>
#include <string>
#include <sstream>

#include "Genten_DiscreteCDF.hpp"
#include "Genten_FacTestSetGenerator.hpp"
#include "Genten_IndxArray.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_RandomMT.hpp"
#include "Genten_Sptensor.hpp"

using namespace std;


//-----------------------------------------------------------------------------
//  Constructor
//-----------------------------------------------------------------------------
Genten::FacTestSetGenerator::FacTestSetGenerator (void)
{
  return;
}


//-----------------------------------------------------------------------------
//  Destructor
//-----------------------------------------------------------------------------
Genten::FacTestSetGenerator::~FacTestSetGenerator (void)
{
  return;
}


//-----------------------------------------------------------------------------
//  Private method
//-----------------------------------------------------------------------------
/**
 *  Return true if the first argument is less than the second.
 */
static ttb_indx  nKEY_LENGTH = 3;
static bool  compareIntArrays (const int * const  parg1,
                               const int * const  parg2)
{
  for (ttb_indx  i = 0; i < nKEY_LENGTH; i++)
  {
    if (parg1[i] < parg2[i])
      return( true );
    else if (parg1[i] > parg2[i])
      return( false );
  }
  return( false );
}


//-----------------------------------------------------------------------------
//  Private method
//-----------------------------------------------------------------------------
/**
 *  Based on Matlab create_problem.generate_data_sparse.
 */
static bool  drawNonzeroElements (const Genten::IndxArray &  cDims,
                                  const Genten::Ktensor   &  cFactors,
                                  const ttb_indx          nMaxNnz,
                                  Genten::RandomMT  &  cRNG,
                                  Genten::Sptensor  &  cDataTensor)
{
  // Use probabilities in the stochastic factors to choose nMaxNnz sparse
  // tensor elements.  Matlab code first decides how many are in each
  // component, based on the weights as probabilities.  Then for each
  // component, the Matlab code chooses elements in each dimension.
  // Duplicates are counted.

  ttb_indx  nNumComps = cFactors.ncomponents();

  // Build a histogram for component probabilities.
  Genten::DiscreteCDF  cCompProbs;
  if (cCompProbs.load (cFactors.weights()) == false)
  {
    cout << "*** Failed to load CDF for weights\n";
    return( false );
  }

  int *  naCompCounts = new int[nNumComps * sizeof(int)];
  for (ttb_indx  c = 0; c < nNumComps; c++)
    naCompCounts[c] = 0;

  // Sample and total the number of nonzeroes to draw from each component.
  for (ttb_indx  i = 0; i < nMaxNnz; i++)
  {
    ttb_real  dNextRan = cRNG.genMatlabMT();
    naCompCounts[cCompProbs.getRandomSample (dNextRan)]++;
  }

  int  nLargestCompCount = 0;
  for (ttb_indx  c = 0; c < nNumComps; c++)
  {
    if (naCompCounts[c] > nLargestCompCount)
      nLargestCompCount = naCompCounts[c];
  }

  // Build a CDF for each factor vector, then sample it for nonzero indices.
  // Count the nonzeroes in a set, incrementing when duplicates are found.
  Genten::DiscreteCDF *  pCDFs
    = new Genten::DiscreteCDF[cDims.size() * sizeof(Genten::DiscreteCDF *)];
  nKEY_LENGTH = cDims.size();
  map<int *,int, bool(*)(const int * const, const int * const)>
    cIndexCounts (compareIntArrays);
  map<int *,int, bool(*)(const int * const, const int * const)>::iterator  it;

  // g++ compiler requires this syntax, instead of "new (int *)[...]".
  int **  pTempIndices = new int*[ cDims.size() * sizeof(int *) ];
  for (ttb_indx  n = 0; n < cDims.size(); n++)
    pTempIndices[n] = new int[ nLargestCompCount * sizeof(int) ];

  // Loop ordering matches Matlab generate_data_sparse.
  for (ttb_indx  c = 0; c < nNumComps; c++)
  {
    for (ttb_indx  n = 0; n < cDims.size(); n++)
    {
      if (pCDFs[n].load (cFactors[n], c) == false)
      {
        cout << "*** Failed to load CDF for component " << c
             << ", dim " << n << "\n";
        return( false );
      }
    }

    // Could avoid storing all the indices by reordering these two
    // loops, but have to match the Matlab code since random numbers
    // must be generated in the same sequence.
    for (ttb_indx  n = 0; n < cDims.size(); n++)
    {
      for (int  i = 0; i < naCompCounts[c]; i++)
      {
        ttb_real  dNextRan = cRNG.genMatlabMT();
        pTempIndices[n][i] = (int) pCDFs[n].getRandomSample (dNextRan);
      }
    }

    // Store the new nonzeroes, counting duplicates.
    // I suspect this is what makes execution slow for large factors.
    // Making the key into a string is definitely slower.
    for (int  i = 0; i < naCompCounts[c]; i++)
    {
      int *  naNextKey = new int[nKEY_LENGTH * sizeof(int) ];
      for (ttb_indx  n = 0; n < cDims.size(); n++)
        naNextKey[n] = pTempIndices[n][i];
      it = cIndexCounts.find (naNextKey);
      if (it == cIndexCounts.end())
      {
        cIndexCounts.insert (pair<int *,int>(naNextKey, 1));
      }
      else
      {
        (*it).second += 1;
        delete[]  naNextKey;
      }
    }
  }

  // Form the sparse data tensor from the index counts.
  cDataTensor = Genten::Sptensor(cDims, cIndexCounts.size());
  ttb_indx  nNextIndex = 0;
  for (it = cIndexCounts.begin(); it != cIndexCounts.end(); it++)
  {
    int *  naTmp = (*it).first;
    for (ttb_indx  n = 0; n < cDims.size(); n++)
    {
      cDataTensor.subscript (nNextIndex, n) = naTmp[n];
    }
    cDataTensor.value(nNextIndex) = (*it).second;

    delete[] (*it).first;
    nNextIndex++;
  }

  for (ttb_indx  n = 0; n < cDims.size(); n++)
    delete[] pTempIndices[n];
  delete[] pTempIndices;
  delete[] pCDFs;
  delete[] naCompCounts;

  return( true );
}


//-----------------------------------------------------------------------------
//  Public method
//-----------------------------------------------------------------------------
bool  Genten::FacTestSetGenerator::genSpFromRndKtensor(
  const IndxArray &  cDims,
  const ttb_indx     nNumComps,
  const ttb_indx     nMaxNnz,
  RandomMT  &  cRNG,
  Sptensor  &  cDataTensor,
  Ktensor   &  cExpectedFactors) const
{
  // Check the arguments.
  if (nNumComps <= 0)
  {
    cout << "*** Value for nNumComps must be positive\n";
    return( false );
  }
  if (nMaxNnz <= 0)
  {
    cout << "*** Value for nMaxNnz must be positive\n";
    return( false );
  }

  // Set the expected factors to random stochastic values, using
  // the same random samples as Matlab.
  cExpectedFactors = Genten::Ktensor (nNumComps, cDims.size(), cDims);
  cExpectedFactors.setRandomUniform (true, cRNG);

  // Generate nonzero elements in the data tensor.
  return( drawNonzeroElements(cDims,
                              cExpectedFactors,
                              nMaxNnz,
                              cRNG,
                              cDataTensor) );
}

//-----------------------------------------------------------------------------
//  Public method
//-----------------------------------------------------------------------------
void  Genten::FacTestSetGenerator::genDnFromRndKtensor(
  const IndxArray &  cDims,
  const ttb_indx     nNumComps,
  RandomMT  &  cRNG,
  Tensor    &  cDataTensor,
  Ktensor   &  cExpectedFactors) const
{
  // Check the arguments.
  if (nNumComps <= 0)
  {
    Genten::error("*** Value for nNumComps must be positive\n");
  }

  // Set the expected factors to random stochastic values, using
  // the same random samples as Matlab.
  cExpectedFactors = Genten::Ktensor (nNumComps, cDims.size(), cDims);
  cExpectedFactors.setRandomUniform (true, cRNG);

  // Generate the data tensor.
  cDataTensor = Tensor(cExpectedFactors);
}


//-----------------------------------------------------------------------------
//  Public method
//-----------------------------------------------------------------------------

typedef std::pair<double,int> SortablePair;
bool SortablePairComparator (const SortablePair &  a,
                             const SortablePair &  b)
{
  return( a.first < b.first );
}

bool  Genten::FacTestSetGenerator::genSpFromBoostedRndKtensor
(const IndxArray &  cDims,
 const ttb_indx     nNumComps,
 const ttb_indx     nMaxNnz,
 const ttb_real     dFracBoosted,
 const ttb_real     dMaxValue,
 const ttb_real     dSmallValue,
 RandomMT  &  cRNG,
 Sptensor  &  cDataTensor,
 Ktensor   &  cExpectedFactors) const
{
  // Check the arguments.
  if (nNumComps <= 0)
  {
    cout << "*** Value for nNumComps must be positive\n";
    return( false );
  }
  if (nMaxNnz <= 0)
  {
    cout << "*** Value for nMaxNnz must be positive\n";
    return( false );
  }
  if ((dFracBoosted < 0.0) || (dFracBoosted > 1.0))
  {
    cout << "*** Value for dFracBoosted must be in the range [0,1]\n";
    return( false );
  }
  if (dMaxValue < 1.0)
  {
    cout << "*** Value for dMaxValue cannot be less than one\n";
    return( false );
  }

  // Set the expected factors to random stochastic values, using
  // the same random samples as Matlab.
  cExpectedFactors = Ktensor(nNumComps, cDims.size(), cDims);
  cExpectedFactors.setWeights(0.0);
  cExpectedFactors.setMatrices(0.0);
  for (ttb_indx  n = 0; n < cDims.size(); n++)
  {
    double *  daV = new double[cDims[n]];
    for (ttb_indx  i = 0; i < cDims[n]; i++)
    {
      daV[i] = dSmallValue;
    }
    int  nNumToBoost = (int) (dFracBoosted * ((double) cDims[n]));
    if (nNumToBoost > (int) cDims[n])
      nNumToBoost = (int) cDims[n];
    for (ttb_indx  r = 0; r < nNumComps; r++)
    {
      for (int  i = 0; i < nNumToBoost; i++)
      {
        daV[i] = 1.0 + (dMaxValue - 1.0) * cRNG.genMatlabMT();
      }

      // Randomly permute the factor vector (randperm in Matlab).
      vector<SortablePair>  caP;
      for (ttb_indx  i = 0; i < cDims[n]; i++)
      {
        double  d = cRNG.genMatlabMT();
        caP.push_back (SortablePair(d,i));
      }
      sort(caP.begin(), caP.end(), SortablePairComparator);
      for (ttb_indx  i = 0; i < cDims[n]; i++)
      {
        cExpectedFactors[n].entry(i,r) = daV[caP[i].second];
      }
    }
    delete[]  daV;
  }

  // Choose random component weights and normalize to make the factor
  // matrix completely stochastic.
  for (ttb_indx  r = 0; r < nNumComps; r++)
  {
    cExpectedFactors.weights()[r] = cRNG.genMatlabMT();
  }
  cExpectedFactors.normalize(Genten::NormOne);
  double  dTotalWeight = 0.0;
  for (ttb_indx  r = 0; r < nNumComps; r++)
  {
    dTotalWeight += cExpectedFactors.weights()[r];
  }
  for (ttb_indx  r = 0; r < nNumComps; r++)
  {
    cExpectedFactors.weights()[r]
      = cExpectedFactors.weights()[r] / dTotalWeight;
  }

  // Generate nonzero elements in the data tensor.
  if (drawNonzeroElements(cDims,
                          cExpectedFactors,
                          nMaxNnz,
                          cRNG,
                          cDataTensor) == false)
  {
    return( false );
  }

  // Rescale the weights so the expected factors sum to the
  // target number of samples.
  for (ttb_indx  r = 0; r < nNumComps; r++)
  {
    cExpectedFactors.weights()[r] *= nMaxNnz;
  }

  return( true );
}
