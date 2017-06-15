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

/*!
  @file Genten_MixedFormatOps.h
  @brief Methods that perform operations between objects of mixed format.
*/

#pragma once

#include "Genten_FacMatrix.h"
#include "Genten_Ktensor.h"
#include "Genten_Sptensor.h"
#include "Genten_Sptensor_perm.h"
#include "Genten_Sptensor_row.h"
#include "Genten_Util.h"

namespace Genten
{
  //---- Methods for innerprod.

  // Inner product between a sparse tensor and a Ktensor.
  /* Compute the element-wise dot product of all elements.
   */
  ttb_real innerprod(const Genten::Sptensor & s,
                     const Genten::Ktensor  & u);

  // Inner product between a sparse tensor and a Ktensor with weights.
  /* Compute the element-wise dot product of all elements, using an
   * alternate weight vector for the Ktensor.  Used by CpAls, which
   * separates the weights from the Ktensor while iterating.
   */
  ttb_real innerprod(const Genten::Sptensor & s,
                     Genten::Ktensor  & u,
                     const Genten::Array    & lambda);


  //---- Methods for mttkrp.

  // Matricized sparse tensor times Khatri-Rao product.
  /* Matricizes the Sptensor X along mode n, and computes the product
     of this with the factor matrices of Ktensor u, excluding mode n.
     Modes are indexed starting with 0.
     The result is scaled by the lambda weights and placed into u[n];
     hence, u[n] is not normalized.

     More specifically, the operation forms X_n * W_n, where X_n is the
     mode n matricization of X, and W_n is the cumulative Khatri-Rao product
     of all factor matrices in u except the nth.
  */
  void mttkrp(const Genten::Sptensor & X,
              Genten::Ktensor  & u,
              ttb_indx        n);
  void mttkrp(const Genten::Sptensor_perm  & X,
              Genten::Ktensor & u,
              ttb_indx              n);
  void mttkrp(const Genten::Sptensor_row   & X,
              Genten::Ktensor & u,
              ttb_indx              n);

  // Matricized sparse tensor times Khatri-Rao product.
  /* Same as above except that rather than overwrite u[n],
     the answer is put into v. */
  void mttkrp(const Genten::Sptensor  & X,
              const Genten::Ktensor   & u,
              ttb_indx         n,
              Genten::FacMatrix & v);
  void mttkrp(const Genten::Sptensor_perm    & X,
              const Genten::Ktensor   & u,
              ttb_indx                n,
              Genten::FacMatrix & v);
  void mttkrp(const Genten::Sptensor_row     & X,
              const Genten::Ktensor   & u,
              ttb_indx                n,
              Genten::FacMatrix & v);

}     //-- namespace Genten
