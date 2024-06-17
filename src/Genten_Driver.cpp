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
#include "Teuchos_RCP.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#endif
#ifdef HAVE_LBFGSB
#include "Genten_CP_Opt_Lbfgsb.hpp"
#endif
#ifdef HAVE_GCP
#include "Genten_GCP_LossFunctions.hpp"
#include "Genten_GCP_SGD.hpp"
#include "Genten_GCP_SGD_SA.hpp"
#ifdef HAVE_DIST
#include "Genten_GCP_FedOpt.hpp"
#endif
#ifdef HAVE_LBFGSB
#include "Genten_GCP_Opt_Lbfgsb.hpp"
#endif
#ifdef HAVE_ROL
#include "Genten_GCP_Opt_Rol.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#endif
#endif
#ifdef HAVE_TEUCHOS
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_StackedTimer.hpp"
#endif

namespace Genten {

template <typename ExecSpace>
void print_environment(const SptensorT<ExecSpace>& x,
                       const DistTensorContext<ExecSpace>& dtc,
                       std::ostream& out)
{
  const ttb_indx nd = x.ndims();
  const ttb_indx nnz = dtc.globalNNZ(x);
  const ttb_real tsz = dtc.globalNumelFloat(x);
  const ttb_real nz = tsz - nnz;
  const ttb_real nrm = dtc.globalNorm(x);
  if (DistContext::rank() == 0) {
    out << std::endl
        << "Sparse tensor: " << std::endl << "  ";
    for (ttb_indx i=0; i<nd; ++i) {
      out << dtc.dims()[i] << " ";
      if (i<nd-1)
        out << "x ";
    }
    out << "(" << tsz << " total entries)" << std::endl
        << "  " << nnz << " ("
        << std::setprecision(1) << std::fixed << 100.0*(nnz/tsz)
        << "%) Nonzeros" << " and "
        << std::setprecision(0) << std::fixed << nz << " ("
        << std::setprecision(1) << std::fixed << 100.0*(nz/tsz)
        << "%) Zeros" << std::endl
        << "  " << std::setprecision(1) << std::scientific << nrm
        << " Frobenius norm" << std::endl << std::endl
        << "Execution environment:" << std::endl
        << "  MPI grid: ";
    for (ttb_indx i=0; i<nd; ++i) {
      out << dtc.pmap().gridDim(i) << " ";
      if (i<nd-1)
        out << "x ";
    }
    out << "processes (" << dtc.nprocs() << " total)" << std::endl
        << "  Execution space: "
        << SpaceProperties<ExecSpace>::verbose_name()
        << std::endl;
  }
}

template <typename ExecSpace>
void print_environment(const TensorT<ExecSpace>& x,
                       const DistTensorContext<ExecSpace>& dtc,
                       std::ostream& out)
{
  const ttb_indx nd = x.ndims();
  const ttb_real tsz = dtc.globalNumelFloat(x);
  const ttb_real nrm = dtc.globalNorm(x);
  if (DistContext::rank() == 0) {
    out << std::endl
        << "Dense tensor: " << std::endl << "  ";
    for (ttb_indx i=0; i<nd; ++i) {
      out << dtc.dims()[i] << " ";
      if (i<nd-1)
        out << "x ";
    }
    out << "(" << tsz << " total entries)" << std::endl
        << "  " << std::setprecision(1) << std::scientific << nrm
        << " Frobenius norm" << std::endl << std::endl
        << "Execution environment:" << std::endl
        << "  MPI grid: ";
    for (ttb_indx i=0; i<nd; ++i) {
      out << dtc.pmap().gridDim(i) << " ";
      if (i<nd-1)
        out << "x ";
    }
    out << "processes (" << dtc.nprocs() << " total)" << std::endl
        << "  Execution space: "
        << SpaceProperties<ExecSpace>::verbose_name()
        << std::endl;
  }
}

template<typename ExecSpace>
KtensorT<ExecSpace>
driver(const DistTensorContext<ExecSpace>& dtc,
       SptensorT<ExecSpace>& x,
       KtensorT<ExecSpace>& u,
       AlgParams& algParams,
       PerfHistory& history,
       std::ostream& out_in)
{
  GENTEN_TIME_MONITOR("Genten driver");

  typedef Genten::KtensorT<ExecSpace> Ktensor_type;
  typedef Genten::KtensorT<Genten::DefaultHostExecutionSpace> Ktensor_host_type;

  // Set parallel output stream
  const ProcessorMap* pmap = dtc.pmap_ptr().get();
  std::ostream& out = pmap->gridRank() == 0 ? out_in : Genten::bhcout;

  Genten::SystemTimer timer(3, algParams.timings, pmap);

  //out.setf(std::ios_base::scientific);
  out.precision(2);

  // Generate a random starting point if initial guess is empty
  if (u.ncomponents() == 0 && u.ndims() == 0) {
    timer.start(0);
    u = dtc.randomInitialGuess(x, algParams.rank, algParams.seed,
                               algParams.prng,
                               algParams.scale_guess_by_norm_x,
                               algParams.dist_guess_method);
    timer.stop(0);
    if (algParams.timings)
      out << "\nCreating random initial guess took " << timer.getTotalTime(0)
          << " seconds\n";
  }

  if (algParams.debug) {
    Ktensor_host_type u0 =
      dtc.template importToRoot<Genten::DefaultHostExecutionSpace>(u);
    Genten::print_ktensor(u0, out, "Initial guess");
  }

  // Fixup algorithmic choices
  algParams.sparse = true;
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
  if (((algParams.mttkrp_method == Genten::MTTKRP_Method::Perm &&
        (algParams.method == Genten::Solver_Method::CP_ALS ||
         algParams.mttkrp_all_method == Genten::MTTKRP_All_Method::Iterated)) ||
       (algParams.hess_vec_tensor_method == Genten::Hess_Vec_Tensor_Method::Perm &&
        algParams.method == Genten::Solver_Method::CP_OPT &&
        algParams.opt_method == Genten::Opt_Method::ROL &&
        algParams.hess_vec_method == Genten::Hess_Vec_Method::Full)) &&
      !x.havePerm()) {
    timer.start(1);
    x.createPermutation();
    timer.stop(1);
    if (algParams.timings)
      out << "Creating permutation arrays for perm MTTKRP/hess-vec method took " << timer.getTotalTime(1)
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
    if (algParams.dist_update_method != Dist_Update_Method::AllReduce)
      Genten::error("Fused-SA GCP-SGD method requires AllReduce distributed parallelism");
    // Run GCP-SGD
    ttb_indx iter;
    ttb_real resNorm;
    gcp_sgd_sa(x, u, algParams, iter, resNorm, history, out);
  }
  else if (algParams.method == Genten::Solver_Method::GCP_FED) {
#ifdef HAVE_DIST
    x.setProcessorMap(nullptr); // GCP_FedOpt handles communication itself
    u.setProcessorMap(nullptr);
    gcp_fed(dtc, x, u, algParams, history);
#else
    Genten::error("gcp-fed requires MPI support!");
#endif
  }
  else if (algParams.method == Genten::Solver_Method::GCP_OPT) {
    Genten::error("gcp-opt is not implemented for sparse tensors since the gradient evaluation involves a dense MTTKRP.  Try \"gcp-sgd\" instead or convert your tensor to dense using the \"convert_tensor\" utility.");
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
  else {
    Genten::error(std::string("Unknown decomposition method:  ") +
                  Genten::Solver_Method::names[algParams.method]);
  }

  if (algParams.debug) {
    Ktensor_host_type u0 =
      dtc.template importToRoot<Genten::DefaultHostExecutionSpace>(u);
    Genten::print_ktensor(u0, out, "Solution");
  }

#if defined(HAVE_TEUCHOS)
  if (algParams.timings) {
    Teuchos::StackedTimer::OutputOptions options;
    options.output_fraction = true;
    options.output_minmax   = true;
    options.align_columns   = true;
    options.print_warnings  = false;
#ifdef HAVE_DIST
    auto comm = Teuchos::rcp(new Teuchos::MpiComm<int>(pmap->gridComm()));
#else
    auto comm = Teuchos::createSerialComm<int>();
#endif
    Teuchos::TimeMonitor::getStackedTimer()->report(out, comm, options);
  }
#endif

  if (algParams.timings_xml != "") {
#if defined(HAVE_TEUCHOS)
#ifdef HAVE_DIST
    auto comm = Teuchos::rcp(new Teuchos::MpiComm<int>(pmap->gridComm()));
#else
    auto comm = Teuchos::createSerialComm<int>();
#endif
    out << "Saving timings to " << algParams.timings_xml << std::endl;
    std::ofstream timings(algParams.timings_xml);
    Teuchos::TimeMonitor::getStackedTimer()->reportXML(timings, "", "", comm);
    timings.close();
#else
    Genten::error("Saving timings to XML file requires Trilinos!");
#endif
    }

  x.setProcessorMap(nullptr);
  u.setProcessorMap(nullptr);

  return u;
}

template<typename ExecSpace>
KtensorT<ExecSpace>
driver(const DistTensorContext<ExecSpace>& dtc,
       TensorT<ExecSpace>& x,
       KtensorT<ExecSpace>& u,
       AlgParams& algParams,
       PerfHistory& history,
       std::ostream& out_in)
{
  GENTEN_TIME_MONITOR("Genten driver");

  typedef Genten::KtensorT<ExecSpace> Ktensor_type;
  typedef Genten::KtensorT<Genten::DefaultHostExecutionSpace> Ktensor_host_type;

  // Set parallel output stream
  const ProcessorMap* pmap = dtc.pmap_ptr().get();
  std::ostream& out = pmap->gridRank() == 0 ? out_in : Genten::bhcout;

  Genten::SystemTimer timer(3);

  // out.setf(std::ios_base::scientific);
  out.precision(2);

  // Generate a random starting point if initial guess is empty
  if (u.ncomponents() == 0 && u.ndims() == 0) {
    u = dtc.randomInitialGuess(x, algParams.rank, algParams.seed,
                               algParams.prng,
                               algParams.scale_guess_by_norm_x,
                               algParams.dist_guess_method);
    timer.stop(0);
    if (algParams.timings)
      out << "\nCreating random initial guess took " << timer.getTotalTime(0)
          << " seconds\n";
  }

  if (algParams.debug) {
     Ktensor_host_type u0 =
      dtc.template importToRoot<Genten::DefaultHostExecutionSpace>(u);
     Genten::print_ktensor(u0, out, "Initial guess");
  }

  // Fixup algorithmic choices
  algParams.sparse = false;
  algParams.fixup<ExecSpace>(out);

  // Set parallel maps
  x.setProcessorMap(pmap);
  u.setProcessorMap(pmap);

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
    Genten::error("Fused-SA GCP-SGD method does not work with dense tensors");
  }
  else if (algParams.method == Genten::Solver_Method::GCP_OPT) {
    timer.start(2);
    if (algParams.opt_method == Genten::Opt_Method::LBFGSB) {
#ifdef HAVE_LBFGSB
      // Run GCP-OPT using L-BFGS-B.  It does not support MPI parallelism
      if (dtc.nprocs() > 1)
        Genten::error("GCP-OPT using L-BFGS-B does not support MPI parallelism with > 1 processor.  Try ROL instead.");
      gcp_opt_lbfgsb(x, u, algParams, history);
#else
      Genten::error("L-BFGS-B requested but not available!");
#endif
    }
    else if (algParams.opt_method == Genten::Opt_Method::ROL) {
#ifdef HAVE_ROL
      // Run GCP-OPT using ROL
      Teuchos::RCP<Teuchos::ParameterList> rol_params;
      if (algParams.rolfilename != "")
        rol_params = Teuchos::getParametersFromXmlFile(algParams.rolfilename);
      if (rol_params != Teuchos::null)
        gcp_opt_rol(x, u, algParams, history, *rol_params, out);
      else
        gcp_opt_rol(x, u, algParams, history, out);
#else
      Genten::error("ROL requested but not available!");
#endif
    }
    else
      Genten::error("Invalid opt method!");
    timer.stop(2);
    if (algParams.timings)
      out << "GCP-OPT took " << timer.getTotalTime(2) << " seconds\n";
  }
#endif
  else {
    Genten::error(std::string("Unknown decomposition method:  ") +
                  Genten::Solver_Method::names[algParams.method]);
  }

  if (algParams.debug) {
    Ktensor_host_type u0 =
      dtc.template importToRoot<Genten::DefaultHostExecutionSpace>(u);
    Genten::print_ktensor(u0, out, "Solution");
  }

#if defined(HAVE_TEUCHOS)
  if (algParams.timings) {
    Teuchos::StackedTimer::OutputOptions options;
    options.output_fraction = true;
    options.output_minmax   = true;
    options.align_columns   = true;
    options.print_warnings  = false;
#ifdef HAVE_DIST
    auto comm = Teuchos::rcp(new Teuchos::MpiComm<int>(pmap->gridComm()));
#else
    auto comm = Teuchos::createSerialComm<int>();
#endif
    Teuchos::TimeMonitor::getStackedTimer()->report(out, comm, options);
  }
#endif

  if (algParams.timings_xml != "") {
#if defined(HAVE_TEUCHOS)
#ifdef HAVE_DIST
    auto comm = Teuchos::rcp(new Teuchos::MpiComm<int>(pmap->gridComm()));
#else
    auto comm = Teuchos::createSerialComm<int>();
#endif
    out << "Saving timings to " << algParams.timings_xml << std::endl;
    std::ofstream timings(algParams.timings_xml);
    Teuchos::TimeMonitor::getStackedTimer()->reportXML(timings, "", "", comm);
    timings.close();
#else
    Genten::error("Saving timings to XML file requires Trilinos!");
#endif
    }

  x.setProcessorMap(nullptr);
  u.setProcessorMap(nullptr);

  return u;
}

}

#define INST_MACRO(SPACE)                                               \
  template void print_environment<SPACE>(                               \
    const SptensorT<SPACE>& x,                                          \
    const DistTensorContext<SPACE>& dtc,                                \
    std::ostream& out);                                                 \
                                                                        \
  template void print_environment<SPACE>(                               \
    const TensorT<SPACE>& x,                                            \
    const DistTensorContext<SPACE>& dtc,                                \
    std::ostream& out);                                                 \
                                                                        \
  template KtensorT<SPACE>                                              \
  driver<SPACE>(                                                        \
    const DistTensorContext<SPACE>& dtc,                                \
    SptensorT<SPACE>& x,                                                \
    KtensorT<SPACE>& u_init,                                            \
    AlgParams& algParams,                                               \
    PerfHistory& history,                                               \
    std::ostream& os);                                                  \
                                                                        \
  template KtensorT<SPACE>                                              \
  driver<SPACE>(                                                        \
    const DistTensorContext<SPACE>& dtc,                                \
    TensorT<SPACE>& x,                                                  \
    KtensorT<SPACE>& u_init,                                            \
    AlgParams& algParams,                                               \
    PerfHistory& history,                                               \
    std::ostream& os);

GENTEN_INST(INST_MACRO)
