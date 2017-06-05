//@Header
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

/*!
  @file Genten_DiscreteCDF.h
  @brief Class declaration for the CDF of a discrete random variable.
 */


#ifndef GENTEN_DISCRETECDF_H
#define GENTEN_DISCRETECDF_H

#include "Genten_Array.h"
#include "Genten_FacMatrix.h"

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
