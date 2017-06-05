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
  @file Genten_FacMatArray.h
  @brief Container class for collections of Genten::FacMatrix objects.
 */

#pragma once
#include "Genten_Util.h"
#include "Genten_IndxArray.h"
#include "Genten_FacMatrix.h"

namespace Genten
{

class FacMatArray
{
public:

  // ----- CREATER & DESTROY -----

  // Empty constructor
  KOKKOS_INLINE_FUNCTION
  FacMatArray() = default;

  // Construct an array to hold n factor matrices.
  FacMatArray(ttb_indx n) : data("Genten::FacMatArray::data",n) {}

  // Construct an array to hold n factor matrices.
  FacMatArray(ttb_indx n, const Genten::IndxArray & nrow,
              ttb_indx ncol) :
    data("Genten::FacMatArray::data",n)
  {
    for (ttb_indx i=0; i<n; ++i)
      data[i] = Genten::FacMatrix(nrow[i],ncol);
  }

  // Copy constructor
  KOKKOS_INLINE_FUNCTION
  FacMatArray(const FacMatArray & src) = default;

  // Destructor.
  KOKKOS_INLINE_FUNCTION
  ~FacMatArray() = default;

  // ----- RESIZE & RESET -----

  // Set all entries of all matrices to val
  void operator=(ttb_real val)
  {
    ttb_indx sz = size();
    for (ttb_indx i=0; i<sz; ++i)
      data[i] = val;
  }

  // Make a copy of an existing array.
  KOKKOS_INLINE_FUNCTION
  Genten::FacMatArray & operator=(const Genten::FacMatArray & src) = default;

  // ----- PROPERTIES -----

  // Return the number of factor matrices.
  KOKKOS_INLINE_FUNCTION
  ttb_indx size() const
  {
    return data.dimension_0();
  }

  // count the total ttb_reals currently stored here for any purpose
  ttb_indx reals() const;

  // ----- ELEMENT ACCESS -----

  // Return reference to n-th factor matrix
  KOKKOS_INLINE_FUNCTION
  Genten::FacMatrix & operator[](ttb_indx n) const
  {
    return data[n];
  }

private:

  // Array of factor matrices stored as a 3-D array
  typedef Kokkos::View<Genten::FacMatrix*> view_type;
  view_type data;
};

}
