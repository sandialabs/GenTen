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

#include "Genten_Sptensor.hpp"

Genten::Sptensor::
Sptensor(ttb_indx nd, ttb_real * sz, ttb_indx nz, ttb_real * vls,
         ttb_real * sbs):
  siz(nd,sz), nNumDims(nd), values(nz,vls,false),
  subs("Genten::Sptensor::subs",nz,nd)
{
  // convert subscripts to ttb_indx with zero indexing and transpose subs array
  // to store each nonzero's subscripts contiguously
  for (ttb_indx i = 0; i < nz; i ++)
  {
    for (ttb_indx j = 0; j < nd; j ++)
    {
      subs(i,j) = (ttb_indx) sbs[i+j*nz] - 1;
    }
  }
}

Genten::Sptensor::
Sptensor(ttb_indx nd, ttb_indx *dims, ttb_indx nz, ttb_real *vals,
         ttb_indx *subscripts):
  siz(nd,dims), nNumDims(nd), values(nz,vals,false),
  subs("Genten::Sptensor::subs",nd,nz)
{
  // Copy subscripts into subs.  Because of polymorphic layout, we can't
  // assume subs and subscripts are ordered in the same way
  for (ttb_indx i = 0; i < nz; i ++)
  {
    for (ttb_indx j = 0; j < nd; j ++)
    {
      subs(i,j) = (ttb_indx) subscripts[i*nd+j];
    }
  }
}

Genten::Sptensor::
Sptensor(const std::vector<ttb_indx>& dims,
         const std::vector<ttb_real>& vals,
         const std::vector< std::vector<ttb_indx> >& subscripts):
  siz(ttb_indx(dims.size()),const_cast<ttb_indx*>(dims.data())),
  nNumDims(dims.size()),
  values(vals.size(),const_cast<ttb_real*>(vals.data()),false),
  subs("Genten::Sptensor::subs",vals.size(),dims.size())
{
  for (ttb_indx i = 0; i < vals.size(); i ++)
  {
    for (ttb_indx j = 0; j < dims.size(); j ++)
    {
      subs(i,j) = subscripts[i][j];
    }
  }
}

void Genten::Sptensor::
words(ttb_indx& iw, ttb_indx& rw) const
{
  rw = values.size();
  iw = subs.size() + nNumDims;
}

bool Genten::Sptensor::
isEqual(const Genten::Sptensor & b, ttb_real tol) const
{
  // Check for equal sizes.
  if (this->ndims() != b.ndims())
  {
    return( false );
  }
  for (ttb_indx  i = 0; i < ndims(); i++)
  {
    if (this->size(i) != b.size(i))
    {
      return( false );
    }
  }
  if (this->nnz() != b.nnz())
  {
    return( false );
  }

  // Check that elements are equal.
  for (ttb_indx  i = 0; i < nnz(); i++)
  {
    if (Genten::isEqualToTol(this->value(i), b.value(i), tol) == false)
    {
      return( false );
    }
  }

  return( true );
}

void Genten::Sptensor::
times(const Genten::Ktensor & K, const Genten::Sptensor & X)
{
  // Copy X into this (including its size array)
  deep_copy(X);

  // Check sizes
  assert(K.isConsistent(siz));

  // Stream through nonzeros
  Genten::IndxArray subs(nNumDims);
  ttb_indx nz = this->nnz();
  for (ttb_indx i = 0; i < nz; i ++)
  {
    this->getSubscripts(i,subs);
    values[i] *= K.entry(subs);
  }

  //TODO: Check for any zeros!
}

void Genten::Sptensor::
divide(const Genten::Ktensor & K, const Genten::Sptensor & X, ttb_real epsilon)
{
  // Copy X into this (including its size array)
  deep_copy(X);

  // Check sizes
  assert(K.isConsistent(siz));

  // Stream through nonzeros
  Genten::IndxArray subs(nNumDims);
  ttb_indx nz = this->nnz();
  for (ttb_indx i = 0; i < nz; i ++)
  {
    this->getSubscripts(i,subs);
    ttb_real val = K.entry(subs);
    if (fabs(val) < epsilon)
    {
      values[i] /= epsilon;
    }
    else
    {
      values[i] /= val;
    }
  }
}
