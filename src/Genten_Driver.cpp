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
  @file Genten_CpAls.cpp
  @brief CP-ALS algorithm, in template form to allow different data tensor types.
*/

#include "Genten_CpAls.hpp"

#include "Genten_SystemTimer.hpp"
#include "Genten_MixedFormatOps.hpp"
#include "Genten_GCP_LossFunctions.hpp"
#include "Genten_IOtext.hpp"

#ifdef HAVE_ROL
#include "Genten_GCP_Opt.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_TimeMonitor.hpp"
#endif

namespace Genten {

template<typename ExecSpace>
KtensorT<ExecSpace>
driver(SptensorT<ExecSpace>& x,
       KtensorT<ExecSpace>& u_init,
       const std::string& method,
       const ttb_indx rank,
       const std::string& rolfilename,
       const GCP_LossFunction::type loss_function_type,
       const ttb_real loss_eps,
       const unsigned long seed,
       const bool prng,
       const ttb_indx maxiters,
       const ttb_real tol,
       const ttb_indx printitn,
       const bool debug,
       const bool warmup,
       std::ostream& out,
       AlgParams& algParams)
{
  typedef Genten::SptensorT<ExecSpace> Sptensor_type;
  typedef Genten::SptensorT<Genten::DefaultHostExecutionSpace> Sptensor_host_type;
  typedef Genten::KtensorT<ExecSpace> Ktensor_type;
  typedef Genten::KtensorT<Genten::DefaultHostExecutionSpace> Ktensor_host_type;

  Genten::SystemTimer timer(3);

  out.setf(std::ios_base::scientific);
  out.precision(2);

  Ktensor_type u(rank, x.ndims(), x.size());
  Ktensor_host_type u_host =
    create_mirror_view( Genten::DefaultHostExecutionSpace(), u );

  // Generate a random starting point if initial guess is empty
  if (u_init.ncomponents() == 0 && u_init.ndims() == 0) {
    u_init = Ktensor_type(rank, x.ndims(), x.size());

    // Matlab cp_als always sets the weights to one.
    u_init.setWeights(1.0);
    Genten::RandomMT cRMT(seed);
    timer.start(0);
    if (prng) {
      u_init.setMatricesScatter(false, true, cRMT);
      if (debug) deep_copy( u_host, u_init );
    }
    else {
      u_host.setMatricesScatter(false, false, cRMT);
      deep_copy( u_init, u_host );
    }
    timer.stop(0);
    out << "Creating random initial guess took " << timer.getTotalTime(0)
        << " seconds\n";
  }

  // Copy initial guess into u
  deep_copy(u, u_init);

  if (debug) Genten::print_ktensor(u_host, out, "Initial guess");

  // Compute default MTTKRP method if that is what was chosen
  if (algParams.mttkrp_method == MTTKRP_Method::Default)
    algParams.mttkrp_method = MTTKRP_Method::computeDefault<ExecSpace>();

  if (warmup)
  {
    // Do a pass through the mttkrp to warm up and make sure the tensor
    // is copied to the device before generating any timings.  Use
    // Sptensor mttkrp and do this before createPermutation() so that
    // createPermutation() timings are not polluted by UVM transfers
    Ktensor_type tmp (rank,x.ndims(),x.size());
    Genten::AlgParams ap = algParams;
    ap.mttkrp_method = Genten::MTTKRP_Method::Atomic;
    for (ttb_indx  n = 0; n < x.ndims(); n++)
      Genten::mttkrp(x, u, n, tmp[n], ap);
  }

  // Perform any post-processing (e.g., permutation and row ptr generation)
  if (algParams.mttkrp_method == Genten::MTTKRP_Method::Perm &&
      !x.havePerm()) {
    timer.start(1);
    x.createPermutation();
    timer.stop(1);
    out << "createPermutation() took " << timer.getTotalTime(1)
        << " seconds\n";
  }

  const bool do_cpals =
    method == "CP-ALS" || method == "cp-als" || method == "cpals";
  const bool do_gcp =
    method == "GCP" || method == "gcp" || method == "GCP-OPT" ||
    method == "gcp_opt" || method == "gcp-opt";

  if (do_cpals) {
    // Run CP-ALS
    ttb_indx iter;
    ttb_real resNorm;
    Genten::cpals_core (x, u, tol, maxiters, -1.0, printitn,
                        iter, resNorm, 0, NULL, out, algParams);
  }
#ifdef HAVE_ROL
  else if (do_gcp) {
    // Run GCP
    Teuchos::RCP<Teuchos::ParameterList> rol_params;
    if (rolfilename != "")
      rol_params = Teuchos::getParametersFromXmlFile(rolfilename);
    timer.start(2);
    if (rol_params != Teuchos::null)
      gcp_opt(x, u, loss_function_type, *rol_params, &out, loss_eps,
              algParams);
    else
      gcp_opt(x, u, loss_function_type, tol, maxiters, &out, loss_eps,
              algParams);
    timer.stop(2);
    out << "GCP took " << timer.getTotalTime(2) << " seconds\n";
  }
#endif
  else {
    Genten::error("Unknown decomposition method:  " + method);
  }

  if (debug) Genten::print_ktensor(u_host, out, "Solution");

#ifdef HAVE_ROL
  if (do_gcp)
    Teuchos::TimeMonitor::summarize();
#endif

  return u;
}

}

#define INST_MACRO(SPACE)                                               \
  template KtensorT<SPACE>                                              \
  driver<SPACE>(                                                        \
    SptensorT<SPACE>& x,                                                \
    KtensorT<SPACE>& u_init,                                            \
    const std::string& method,                                          \
    const ttb_indx rank,                                                \
    const std::string& rolfilename,                                     \
    const GCP_LossFunction::type loss_function_type,                    \
    const ttb_real loss_eps,                                            \
    const unsigned long seed,                                           \
    const bool prng,                                                    \
    const ttb_indx maxiters,                                            \
    const ttb_real tol,                                                 \
    const ttb_indx printitn,                                            \
    const bool debug,                                                   \
    const bool warmup,                                                  \
    std::ostream& os,                                                   \
    AlgParams& algParams);

GENTEN_INST(INST_MACRO)
