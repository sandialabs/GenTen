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

#include "Genten_Driver.hpp"
#include "Genten_SystemTimer.hpp"
#include "Genten_MixedFormatOps.hpp"
#include "Genten_IOtext.hpp"

#include "Genten_CpAls.hpp"
#ifdef HAVE_ROL
#include "Genten_CP_Opt_Rol.hpp"
#endif
#ifdef HAVE_LBFGSB
#include "Genten_CP_Opt_Lbfgsb.hpp"
#endif
#ifdef HAVE_GCP
#include "Genten_GCP_LossFunctions.hpp"
#include "Genten_GCP_SGD.hpp"
#include "Genten_GCP_SGD_SA.hpp"
#ifdef HAVE_DIST
#include "Genten_DistGCP.hpp"
#endif
#ifdef HAVE_ROL
#include "Genten_GCP_Opt.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_TimeMonitor.hpp"
#endif
#endif

namespace Genten {

template<typename ExecSpace>
KtensorT<ExecSpace>
driver(const DistTensorContext& dtc,
       SptensorT<ExecSpace>& x,
       KtensorT<ExecSpace>& u,
       AlgParams& algParams,
       const ptree& ptree,
       PerfHistory& history,
       std::ostream& out_in)
{
  typedef Genten::SptensorT<ExecSpace> Sptensor_type;
  typedef Genten::SptensorT<Genten::DefaultHostExecutionSpace> Sptensor_host_type;
  typedef Genten::KtensorT<ExecSpace> Ktensor_type;
  typedef Genten::KtensorT<Genten::DefaultHostExecutionSpace> Ktensor_host_type;

  Genten::SystemTimer timer(3, algParams.timings);

  // Set parallel output stream
  const ProcessorMap* pmap = dtc.pmap_ptr().get();
  std::ostream& out = pmap->gridRank() == 0 ? out_in : Genten::bhcout;

  out.setf(std::ios_base::scientific);
  out.precision(2);

  // Generate a random starting point if initial guess is empty
  if (u.ncomponents() == 0 && u.ndims() == 0) {
    timer.start(0);
    u = dtc.randomInitialGuess<ExecSpace>(x, algParams.rank, algParams.seed,
                                          algParams.prng,
                                          algParams.dist_guess_method);
    timer.stop(0);
    if (algParams.timings)
      out << "Creating random initial guess took " << timer.getTotalTime(0)
          << " seconds\n";
  }

  if (algParams.debug) {
    Ktensor_type u0 = dtc.importToRoot(u);
    Ktensor_host_type u0_host = create_mirror_view(
    Genten::DefaultHostExecutionSpace(), u0 );
    deep_copy(u0_host, u0);
    Genten::print_ktensor(u0_host, out, "Initial guess");
  }

  // Fixup algorithmic choices
  algParams.fixup<ExecSpace>(out);

  // Set parallel maps
  x.setProcessorMap(pmap);
  u.setProcessorMap(pmap);

  if (algParams.warmup)
  {
    // Do a pass through the mttkrp to warm up and make sure the tensor
    // is copied to the device before generating any timings.  Use
    // Sptensor mttkrp and do this before createPermutation() so that
    // createPermutation() timings are not polluted by UVM transfers
    Ktensor_type tmp (algParams.rank,x.ndims(),x.size());
    Genten::AlgParams ap = algParams;
    ap.mttkrp_method = Genten::MTTKRP_Method::Atomic;
    for (ttb_indx  n = 0; n < x.ndims(); n++)
      Genten::mttkrp(x, u, n, tmp[n], ap);
  }

  // Perform any post-processing (e.g., permutation and row ptr generation)
  if (algParams.mttkrp_method == Genten::MTTKRP_Method::Perm &&
      (algParams.method == Genten::Solver_Method::CP_ALS ||
       algParams.mttkrp_all_method == Genten::MTTKRP_All_Method::Iterated) &&
      !x.havePerm()) {
    timer.start(1);
    x.createPermutation();
    timer.stop(1);
    if (algParams.timings)
      out << "Creating permutation arrays for perm MTTKRP method took " << timer.getTotalTime(1)
          << " seconds\n";
  }

  if (algParams.method == Genten::Solver_Method::CP_ALS) {
    // Run CP-ALS
    ttb_indx iter;
    ttb_real resNorm;
    cpals_core(x, u, algParams, iter, resNorm, 1, history, out);
  }
  else if (algParams.method == Genten::Solver_Method::CP_OPT) {
    timer.start(2);
    if (algParams.opt_method == Genten::Opt_Method::LBFGSB) {
#ifdef HAVE_LBFGSB
      // Run CP-OPT using L-BFGS-B.  It does not support MPI parallelism
      if (dtc.nprocs() > 1)
        Genten::error("CP-OPT using L-BFGS-B does not support MPI parallelism with > 1 processor.  Try ROL instead.");
      cp_opt_lbfgsb(x, u, algParams, history);
#else
      Genten::error("L-BFGS-B requested but not available!");
#endif
    }
    else if (algParams.opt_method == Genten::Opt_Method::ROL) {
#ifdef HAVE_ROL
      // Run CP-OPT using ROL
      Teuchos::RCP<Teuchos::ParameterList> rol_params;
      if (algParams.rolfilename != "")
        rol_params = Teuchos::getParametersFromXmlFile(algParams.rolfilename);
      if (rol_params != Teuchos::null)
        cp_opt_rol(x, u, algParams, history, *rol_params, out);
      else
        cp_opt_rol(x, u, algParams, history, out);
#else
      Genten::error("ROL requested but not available!");
#endif
    }
    else
      Genten::error("Invalid opt method!");
    timer.stop(2);
    if (algParams.timings)
      out << "CP-OPT took " << timer.getTotalTime(2) << " seconds\n";
  }
#ifdef HAVE_GCP
  else if (algParams.method == Genten::Solver_Method::GCP_SGD &&
           !algParams.fuse_sa) {
    // Run GCP-SGD
    ttb_indx iter;
    ttb_real resNorm;
    gcp_sgd(x, u, algParams, iter, resNorm, history, out);
  }
  else if (algParams.method == Genten::Solver_Method::GCP_SGD &&
           algParams.fuse_sa) {
    // Run GCP-SGD
    ttb_indx iter;
    ttb_real resNorm;
    gcp_sgd_sa(x, u, algParams, iter, resNorm, history, out);
  }
  else if (algParams.method == Genten::Solver_Method::GCP_SGD_DIST) {
#ifdef HAVE_DIST
    // Run Drew's distributed GCP-SGD implementation
    x.setProcessorMap(nullptr); // DistGCP handles communication itself
    u.setProcessorMap(nullptr);
    DistGCP<ExecSpace> dgcp(dtc, x, u, ptree, history);
    ttb_real resNorm = dgcp.compute();
#else
    Genten::error("gcp-sgd-dist requires MPI support!");
#endif
  }
#ifdef HAVE_ROL
  else if (algParams.method == Genten::Solver_Method::GCP_OPT) {
    Genten::error("gcp-opt is disabled because it doesn't work!");
    // // Run GCP
    // Teuchos::RCP<Teuchos::ParameterList> rol_params;
    // if (algParams.rolfilename != "")
    //   rol_params = Teuchos::getParametersFromXmlFile(algParams.rolfilename);
    // timer.start(2);
    // if (rol_params != Teuchos::null)
    //   gcp_opt(x, u, algParams, *rol_params, &out);
    // else
    //   gcp_opt(x, u, algParams, &out);
    // timer.stop(2);
    // out << "GCP took " << timer.getTotalTime(2) << " seconds\n";
  }
#endif
#endif
  else {
    Genten::error(std::string("Unknown decomposition method:  ") +
                  Genten::Solver_Method::names[algParams.method]);
  }

  if (algParams.debug) {
    Ktensor_type u0 = dtc.importToRoot(u);
    Ktensor_host_type u0_host = create_mirror_view(
      Genten::DefaultHostExecutionSpace(), u0 );
    deep_copy(u0_host, u0);
    Genten::print_ktensor(u0_host, out, "Solution");
  }

#if defined(HAVE_GCP) && defined(HAVE_ROL)
  if (algParams.method == Genten::Solver_Method::GCP_OPT)
    Teuchos::TimeMonitor::summarize();
#endif

  x.setProcessorMap(nullptr);
  u.setProcessorMap(nullptr);

  return u;
}

template<typename ExecSpace>
KtensorT<ExecSpace>
driver(TensorT<ExecSpace>& x,
       KtensorT<ExecSpace>& u_init,
       AlgParams& algParams,
       PerfHistory& history,
       std::ostream& out)
{
  typedef Genten::TensorT<ExecSpace> Tensor_type;
  typedef Genten::TensorT<Genten::DefaultHostExecutionSpace> Tensor_host_type;
  typedef Genten::KtensorT<ExecSpace> Ktensor_type;
  typedef Genten::KtensorT<Genten::DefaultHostExecutionSpace> Ktensor_host_type;

  Genten::SystemTimer timer(3);

  out.setf(std::ios_base::scientific);
  out.precision(2);

  Ktensor_type u(algParams.rank, x.ndims(), x.size());
  Ktensor_host_type u_host =
    create_mirror_view( Genten::DefaultHostExecutionSpace(), u );

  // Generate a random starting point if initial guess is empty
  if (u_init.ncomponents() == 0 && u_init.ndims() == 0) {
    u_init = Ktensor_type(algParams.rank, x.ndims(), x.size());

    Genten::RandomMT cRMT(algParams.seed);
    timer.start(0);
    if (algParams.prng) {
      u_init.setWeights(1.0); // Matlab cp_als always sets the weights to one.
      u_init.setMatricesScatter(false, true, cRMT);
      if (algParams.debug) deep_copy( u_host, u_init );
    }
    else {
      u_host.setWeights(1.0); // Matlab cp_als always sets the weights to one.
      u_host.setMatricesScatter(false, false, cRMT);
      deep_copy( u_init, u_host );
    }
    // Normalize
    const ttb_real norm_x = x.norm();
    const ttb_real norm_u = std::sqrt(u_init.normFsq());
    u_init.weights().times(norm_x/norm_u);
    timer.stop(0);
    out << "Creating random initial guess took " << timer.getTotalTime(0)
        << " seconds\n";
  }

  // Copy initial guess into u
  deep_copy(u, u_init);

  if (algParams.debug) Genten::print_ktensor(u_host, out, "Initial guess");

  // Fixup algorithmic choices
  algParams.fixup<ExecSpace>(out);

  if (algParams.warmup)
  {
    // Do a pass through the mttkrp to warm up and make sure the tensor
    // is copied to the device before generating any timings.  Use
    // Tensor mttkrp and do this before createPermutation() so that
    // createPermutation() timings are not polluted by UVM transfers
    Ktensor_type tmp (algParams.rank,x.ndims(),x.size());
    Genten::AlgParams ap = algParams;
    ap.mttkrp_method = Genten::MTTKRP_Method::Atomic;
    for (ttb_indx  n = 0; n < x.ndims(); n++)
      Genten::mttkrp(x, u, n, tmp[n], ap);
  }

  if (algParams.method == Genten::Solver_Method::CP_ALS) {
    // Run CP-ALS
    ttb_indx iter;
    ttb_real resNorm;
    cpals_core(x, u, algParams, iter, resNorm, 1, history, out);
  }
  else if (algParams.method == Genten::Solver_Method::CP_OPT) {
    timer.start(2);
    if (algParams.opt_method == Genten::Opt_Method::LBFGSB) {
#ifdef HAVE_LBFGSB
      // Run CP-OPT using L-BFGS-B
      cp_opt_lbfgsb(x, u, algParams, history);
#else
      Genten::error("L-BFGS-B requested but not available!");
#endif
    }
    else if (algParams.opt_method == Genten::Opt_Method::ROL) {
#ifdef HAVE_ROL
      // Run CP-OPT using ROL
      Teuchos::RCP<Teuchos::ParameterList> rol_params;
      if (algParams.rolfilename != "")
        rol_params = Teuchos::getParametersFromXmlFile(algParams.rolfilename);
      if (rol_params != Teuchos::null)
        cp_opt_rol(x, u, algParams, history, *rol_params, out);
      else
        cp_opt_rol(x, u, algParams, history, out);
#else
      Genten::error("ROL requested but not available!");
#endif
    }
    else
      Genten::error("Invalid opt method!");
    timer.stop(2);
    if (algParams.timings)
      out << "CP-OPT took " << timer.getTotalTime(2) << " seconds\n";
  }
  else {
    Genten::error(std::string("Unknown decomposition method:  ") +
                  Genten::Solver_Method::names[algParams.method]);
  }

  if (algParams.debug) Genten::print_ktensor(u_host, out, "Solution");

  return u;
}

}

#define INST_MACRO(SPACE)                                               \
  template KtensorT<SPACE>                                              \
  driver<SPACE>(                                                        \
    const DistTensorContext& dtc,                                       \
    SptensorT<SPACE>& x,                                                \
    KtensorT<SPACE>& u_init,                                            \
    AlgParams& algParams,                                               \
    const ptree& ptree,                                                 \
    PerfHistory& history,                                               \
    std::ostream& os);                                                  \
                                                                        \
  template KtensorT<SPACE>                                              \
  driver<SPACE>(                                                        \
    TensorT<SPACE>& x,                                                  \
    KtensorT<SPACE>& u_init,                                            \
    AlgParams& algParams,                                               \
    PerfHistory& history,                                               \
    std::ostream& os);

GENTEN_INST(INST_MACRO)
