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

#include "Genten_GCP_Opt.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_RolBoundConstraint.hpp"
#include "Genten_GCP_RolObjective.hpp"
#include "Genten_GCP_LossFunctions.hpp"

#include "ROL_OptimizationProblem.hpp"
#include "ROL_OptimizationSolver.hpp"

#include "Teuchos_TimeMonitor.hpp"

#ifdef HAVE_CALIPER
#include <caliper/cali.h>
#endif

namespace Genten {

  namespace Impl {

    template<typename TensorT, typename ExecSpace, typename LossFunction>
    void gcp_opt_impl(const TensorT& x, KtensorT<ExecSpace>& u,
                      const LossFunction& loss_func,
                      Teuchos::ParameterList& params,
                      std::ostream* stream,
                      const AlgParams& algParams)
    {
      // Create ROL interface
      typedef Genten::GCP_RolObjective<TensorT, LossFunction> objective_type;
      typedef typename objective_type::vector_type vector_type;
      ROL::Ptr<objective_type> objective =
        ROL::makePtr<objective_type>(x, u, loss_func, algParams);
      ROL::Ptr<vector_type> z = objective->createDesignVector();
      z->copyFromKtensor(u);

      // Check objective gradient if requested
      Teuchos::ParameterList& fd_check_pl =
        params.sublist("Finite Difference Check");
      const bool do_fd_check = fd_check_pl.get("Check Gradient", false);
      if (do_fd_check) {
        const int fdOrder = fd_check_pl.get("Finite Difference Order", 1);
        const int numSteps = fd_check_pl.get("Number of Steps", 9);
        const ttb_real largestStep = fd_check_pl.get("Largest Step Size", 1.e0);
        const ttb_real stepReduction = fd_check_pl.get("Step Reduction Factor",
                                                       1.e-1);
        std::vector<ttb_real> fdSteps(numSteps,largestStep);
        for (int i=1; i<numSteps; ++i)
          fdSteps[i] = stepReduction * fdSteps[i-1];

        // Base point for finite difference calculation
        ROL::Ptr< ROL::Vector<ttb_real> > base = z->clone();
        base->set(*z);

        // Direction -- uniformaly random between 0 and 1
        ROL::Ptr< ROL::Vector<ttb_real> > dir = z->clone();
        std::srand(12345);
        dir->randomize(0.0,1.0);

        // Run f.d. calculation
        objective->checkGradient(*base, *dir, fdSteps, true, *stream, fdOrder);
      }

      // Create constraints
      ROL::Ptr<ROL::BoundConstraint<ttb_real> > bounds;
      if (loss_func.has_lower_bound() || loss_func.has_upper_bound()) {
        ROL::Ptr<vector_type> lower = objective->createDesignVector();
        ROL::Ptr<vector_type> upper = objective->createDesignVector();
        lower->setScalar(loss_func.lower_bound());
        upper->setScalar(loss_func.upper_bound());
        bounds =
          ROL::makePtr<RolBoundConstraint<vector_type> >(lower, upper);
      }

      // Create optimization problem
      ROL::OptimizationProblem<ttb_real> problem(objective, z, bounds);

      // Create ROL optimization solver
      ROL::OptimizationSolver<ttb_real> solver(problem, params);

      // Run GCP
      {
        TEUCHOS_FUNC_TIME_MONITOR("GCP_Optimization");
        if (stream != nullptr)
          solver.solve(*stream);
        else
          solver.solve();
        z->copyToKtensor(u);
      }

      // Normalize Ktensor u
      u.normalize(Genten::NormTwo);
      u.arrange();
    }

  }


  template<typename TensorT, typename ExecSpace>
  void gcp_opt(const TensorT& x, KtensorT<ExecSpace>& u,
               const LOSS_FUNCTION_TYPE loss_function_type,
               Teuchos::ParameterList& params,
               std::ostream* stream,
               const ttb_real loss_eps,
               const AlgParams& algParams)
  {
#ifdef HAVE_CALIPER
    cali::Function cali_func("Genten::gcp_opt");
#endif

    // Check size compatibility of the arguments.
    if (u.isConsistent() == false)
      Genten::error("Genten::gcp_opt - ktensor u is not consistent");
    if (x.ndims() != u.ndims())
      Genten::error("Genten::gcp_opt - u and x have different num dims");
    for (ttb_indx  i = 0; i < x.ndims(); i++)
    {
      if (x.size(i) != u[i].nRows())
        Genten::error("Genten::gcp_opt - u and x have different size");
    }

    // Dispatch implementation based on loss function type
    if (loss_function_type == GAUSSIAN)
      Impl::gcp_opt_impl(x, u, GaussianLossFunction(loss_eps), params, stream,
                         algParams);
    // else if (loss_function_type == RAYLEIGH)
    //   Impl::gcp_opt_impl(x, u, RayleighLossFunction(loss_eps), params, stream,
    //                      algParams);
    // else if (loss_function_type == GAMMA)
    //   Impl::gcp_opt_impl(x, u, GammaLossFunction(loss_eps), params, stream,
    //                      algParams);
    // else if (loss_function_type == BERNOULLI)
    //   Impl::gcp_opt_impl(x, u, BernoulliLossFunction(loss_eps), params, stream,
    //                      algParams);
    // else if (loss_function_type == POISSON)
    //   Impl::gcp_opt_impl(x, u, PoissonLossFunction(loss_eps), params, stream,
    //                      algParams);
    else
       Genten::error("Genten::gcp_opt - unknown loss function");
  }

}

#define INST_MACRO(SPACE)                                               \
  template void gcp_opt<SptensorT<SPACE>,SPACE>(                        \
    const SptensorT<SPACE>& x,                                          \
    KtensorT<SPACE>& u,                                                 \
    const LOSS_FUNCTION_TYPE loss_function_type,                        \
    Teuchos::ParameterList& params,                                     \
    std::ostream* stream,                                               \
    const ttb_real loss_eps,                                            \
    const AlgParams& algParms);

GENTEN_INST(INST_MACRO)
