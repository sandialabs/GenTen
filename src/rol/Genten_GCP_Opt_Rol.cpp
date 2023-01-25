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
  @file Genten_GCP.cpp
  @brief GCP algorithm, in template form to allow different data tensor types.
*/

#include <assert.h>
#include <cstdio>

#include "Genten_GCP_Opt_Rol.hpp"
#include "Genten_RolBoundConstraint.hpp"
#include "Genten_GCP_RolObjective.hpp"
#include "Genten_GCP_LossFunctions.hpp"

#include "Genten_SystemTimer.hpp"

#include "ROL_Problem.hpp"
#include "ROL_Solver.hpp"

#include "Teuchos_TimeMonitor.hpp"

#ifdef HAVE_CALIPER
#include <caliper/cali.h>
#endif

namespace Genten {

template<typename ExecSpace>
void gcp_opt_rol(const TensorT<ExecSpace>& x, KtensorT<ExecSpace>& u,
                 const AlgParams& algParams,
                 PerfHistory& history,
                 Teuchos::ParameterList& params,
                 std::ostream& stream)
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::gcp_opt_rol");
#endif

  // Check size compatibility of the arguments.
  if (u.isConsistent() == false)
    Genten::error("Genten::gcp_opt - ktensor u is not consistent");
  if (x.ndims() != u.ndims())
    Genten::error("Genten::gcp_opt - u and x have different num dims");

  const ProcessorMap* pmap = u.getProcessorMap();

  Genten::SystemTimer timer(1, pmap);
  timer.start(0);

  // Distribute the initial guess to have weights of one since the objective
  // does not include gradients w.r.t. weights
  u.distribute(0);

  // Create ROL interface
  typedef Genten::GCP_RolObjectiveBase<ExecSpace> objective_type;
  typedef typename objective_type::vector_type vector_type;
  ROL::Ptr<objective_type> objective = GCP_createRolObjective(
    x, u, algParams, history);
  ROL::Ptr<vector_type> z = objective->createDesignVector();
  z->copyFromKtensor(u);
  auto g = z->dual().clone();
  g->set(z->dual());

  // Create optimization problem
  ROL::Ptr< ROL::Problem<ttb_real> > problem =
    ROL::makePtr< ROL::Problem<ttb_real> >(objective, z, g);

  // Create constraints
  if (objective->lossFunctionHasLowerBound() ||
      objective->lossFunctionHasUpperBound()) {
    ROL::Ptr<vector_type> lower = objective->createDesignVector();
    ROL::Ptr<vector_type> upper = objective->createDesignVector();
    lower->setScalar(objective->lossFunctionLowerBound());
    upper->setScalar(objective->lossFunctionUpperBound());
    ROL::Ptr<ROL::BoundConstraint<ttb_real> > bounds =
      ROL::makePtr<RolBoundConstraint<vector_type> >(lower, upper);
    problem->addBoundConstraint(bounds);
  }

  if (algParams.printitn > 0) {
    const ttb_indx nc = u.ncomponents();
    stream << std::endl
           << "GCP-OPT (ROL):" << std::endl;
    stream << "  CP Rank: " << nc << std::endl
           << "  function type: " << objective->lossFunctionName() << std::endl;
    if (objective->lossFunctionHasLowerBound())
      stream << "  Lower bound: "
             << std::setprecision(2) << std::scientific
             << objective->lossFunctionLowerBound() << std::endl;
    if (objective->lossFunctionHasUpperBound())
      stream << "  Upper bound: "
             << std::setprecision(2) << std::scientific
             << objective->lossFunctionUpperBound() << std::endl;
    stream << "  Gradient method: "
           << MTTKRP_All_Method::names[algParams.mttkrp_all_method];
    if (algParams.mttkrp_all_method == MTTKRP_All_Method::Iterated)
      stream << " (" << MTTKRP_Method::names[algParams.mttkrp_method] << ")";
    stream << " MTTKRP" << std::endl;
  }

  // Finalize problem
  const bool lumpConstraints = false;
  //const bool printToStream   = algParams.printitn > 0;
  problem->finalize(lumpConstraints, false, stream);

  // Check interface consistency
  const bool do_checks = params.get("Check ROL Interface", false);
  if (do_checks)
    problem->check(true, stream);

  // Create ROL optimization solver
  Teuchos::ParameterList& rol_params = params.sublist("ROL");
  ROL::Solver<ttb_real> solver(problem, rol_params);

  // Run GCP
  {
    GENTEN_TIME_MONITOR("GCP_Optimization");
    solver.solve(stream);
    z->copyToKtensor(u);
  }

  // Normalize Ktensor u
  u.normalize(Genten::NormTwo);
  u.arrange();

  // Set final time in history
  timer.stop(0);
  history.lastEntry().cum_time = timer.getTotalTime(0);

  if (algParams.printitn > 0) {
    if (algParams.compute_fit) {
      stream << "Final fit = " << std::setprecision(3) << std::scientific
             <<  objective->computeFit(u) << std::endl;
    }
    stream << "Total time = " << std::setprecision(2) << std::scientific
           << history.lastEntry().cum_time << std::endl
           << std::endl;
  }

}

}

#define INST_MACRO(SPACE)                                               \
  template void gcp_opt_rol<SPACE>(                                     \
    const TensorT<SPACE>& x,                                            \
    KtensorT<SPACE>& u,                                                 \
    const AlgParams& algParms,                                          \
    PerfHistory& history,                                               \
    Teuchos::ParameterList& params,                                     \
    std::ostream& stream);

GENTEN_INST(INST_MACRO)
