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


#include "Genten_Util.h"
#include "Genten_IndxArray.h"
#include <iostream>

// For vtune
#include <sstream>
#include <sys/types.h>
#include <unistd.h>


void Genten::error(std::string s)
{
  std::cerr << "FATAL ERROR: " << s << std::endl;
  throw s;
}

void Genten::sub2ind(const Genten::IndxArray & siz, const Genten::IndxArray & sub, ttb_indx & ind)
{
  // Get the number of dimensions
  ttb_indx nd = siz.size();

  // Error checking
  if (sub.size() != nd)
  {
    error("Genten::Tensor::sub2ind - Size mismatch");
  }

  // More error checking
  for (ttb_indx i = 0; i < nd; i ++)
  {
    if (sub[i] >= siz[i])
    {
      error("Genten::Tensor::sub2ind - subscript out of range");
    }
  }

  // Calculate the linear index from the subscripts
  ind = 0;
  ttb_indx cumprod = 1;
  for (ttb_indx i = 0; i < nd; i ++)
  {
    ind += sub[i] * cumprod;
    cumprod *= siz[i];
  }
}

void Genten::ind2sub(const IndxArray & siz, ttb_indx ind, IndxArray & sub)
{
  // Get the number of dimensions
  ttb_indx nd = siz.size();

  // Error checking
  if (sub.size() != nd)
  {
    error("Genten::Tensor::ind2sub - Size mismatch");
  }

  // Get the total size
  ttb_indx cumprod = siz.prod();

  // More error checking
  if (ind >= cumprod)
  {
    error("Genten::Tensor::sub2ind - index out of range");
  }

  // Calculate the subscripts from the linear index
  sub = IndxArray(nd);
  ttb_indx sbs;
  for (ttb_indx i = nd; i > 0; i --)
  {
    cumprod = cumprod / siz[i-1];
    sbs = ind / cumprod;
    sub[i-1] = sbs;
    ind = ind - (sbs * cumprod);
  }
}

bool  Genten::isEqualToTol(ttb_real  d1,
                           ttb_real  d2,
                           ttb_real  dTol)
{
  // Numerator = fabs(d1 - d2).
  ttb_real  dDiff = fabs(d1 - d2);

  // Denominator  = max(1, fabs(d1), fabs(d2).
  ttb_real  dAbs1 = fabs(d1);
  ttb_real  dAbs2 = fabs(d2);
  ttb_real  dD = 1.0;
  if ((dAbs1 > 1.0) || (dAbs2 > 1.0))
  {
    if (dAbs1 > dAbs2)
      dD = dAbs1;
    else
      dD = dAbs2;
  }

  // Relative difference.
  ttb_real  dRelDiff = dDiff / dD;

  // Compare the relative difference to the tolerance.
  return( dRelDiff < dTol );
}

char *  Genten::getGentenVersion(void)
{
  return( (char *)("Genten Tensor Toolbox 0.0.0") );
}

// Connect executable to vtune for profiling
void Genten::connect_vtune(const int p_rank) {
  std::stringstream cmd;
  pid_t my_os_pid=getpid();
  const std::string vtune_loc =
    "amplxe-cl";
  const std::string output_dir = "./vtune/vtune.";
  cmd << vtune_loc
      << " -collect hotspots -result-dir " << output_dir << p_rank
      << " -target-pid " << my_os_pid << " &";
  if (p_rank == 0)
    std::cout << cmd.str() << std::endl;
  system(cmd.str().c_str());
  system("sleep 10");
}
