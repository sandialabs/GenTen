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

/*!
  @file Genten_DiscreteCDF.h
  @brief Class declaration for the CDF of a discrete random variable.
 */


#ifndef GENTEN_DISCRETECDF_H
#define GENTEN_DISCRETECDF_H

#include "Genten_Array.hpp"
#include "Genten_FacMatrix.hpp"

namespace Genten
{

//----------------------------------------------------------------------
//! Computes random samples from a CDF.
/*!
 *  Assemble the probability histogram of a discrete valued random variable,
 *  and provide methods for efficient random sampling.
 */
//----------------------------------------------------------------------
class DiscreteCDF
{
  public:

    //! Constructor.
    DiscreteCDF (void);

    //! Destructor.
    ~DiscreteCDF (void);


    //! Load with a PDF.
    /*!
     *  @param[in] cPDF     The vector is a probability density function
     *                      histogram: all values are in [0,1) and sum to 1.
     *  @return  False if an argument is illegal.
     */
    bool  load (const Array &   cPDF);

    //! Load with a PDF.
    /*!
     *  @param[in] cPDF     Each column is a probability density function
     *                      histogram: all values are in [0,1) and sum to 1.
     *  @param[in] nColumn  Column vector to load.
     *  @return  False if an argument is illegal.
     */
    bool  load (const FacMatrix &   cPDF,
                const ttb_indx      nColumn);


    //! Return a value of the discrete random variable.
    /*!
     *  Use the random number to select a bin of the CDF, and return the
     *  index of the bin, starting from zero.  The method encapsulates
     *  a search algorithm with worst case log N retrieval time.
     *
     *  The random number is passed so the caller can control random number
     *  generation.  Generally, it is from a uniform distribution.
     *
     *  @param[in] dRandomNumber  Number in [0,1).
     *  @return  Value of the discrete random variable, starting with 0.
     */
    ttb_indx  getRandomSample (ttb_real  dRandomNumber);


  private:

    //! By design, there is no copy constructor.
    DiscreteCDF (const DiscreteCDF &);
    //! By design, there is no assignment operator.
    DiscreteCDF & operator= (const DiscreteCDF &);

    Array  _cCDF;
};

}          //-- namespace Genten

#endif     //-- GENTEN_DISCRETECDF_H
