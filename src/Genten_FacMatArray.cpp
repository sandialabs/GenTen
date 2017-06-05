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


#include "Genten_FacMatArray.h"

ttb_indx Genten::FacMatArray::
reals() const
{
  ttb_indx s =0;
  ttb_indx sz = this->size();
  for(ttb_indx i = 0; i < sz; i ++)
  {
    s += (*this)[i].reals();
  }

  return(s);
}
