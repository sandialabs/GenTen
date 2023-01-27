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

#include <ostream>
#include <iomanip>
#include <sstream>
#include <cmath>

#include "Genten_Array.hpp"
#include "Genten_CpAls.hpp"
#include "Genten_FacMatrix.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_MixedFormatOps.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_Tensor.hpp"
#include "Genten_SystemTimer.hpp"
#include "Genten_Util.hpp"
#include "Genten_Pmap.hpp"
#include "Genten_DistKtensorUpdate.hpp"

#ifdef HAVE_CALIPER
#include <caliper/cali.h>
#endif

namespace Genten {

  //------------------------------------------------------
  // FORWARD REFERENCES
  //------------------------------------------------------
  static ttb_real computeResNorm(const ttb_real  xNorm,
                                 const ttb_real  mNorm,
                                 const ttb_real  xDotm);


  //------------------------------------------------------
  // PUBLIC FUNCTIONS
  //------------------------------------------------------

  /*
   *  Copied from the header file:
   *
   *  @param[in] x          Data tensor to be fit by the model.
   *  @param[in/out] u      Input contains an initial guess for the factors.
   *                        The size of each mode must match the corresponding
   *                        mode of x, and the number of components determines
   *                        how many will be in the result.
   *                        Output contains resulting factorization Ktensor.
   *  @param[in] tol        Stop tolerance for convergence of "fit function".
   *  @param[in] maxIters   Maximum number of iterations allowed.
   *  @param[in] maxSecs    Maximum execution time allowed (CP-ALS will finish
   *                        the current iteration before exiting).
   *                        If negative, execute without a time limit.
   *  @param[in] printIter  Print progress every n iterations.
   *                        If zero, print nothing.
   *  @param[out] numIters  Number of iterations actually completed.
   *  @param[out] resNorm   Square root of Frobenius norm of the residual.
   *  @param[in] perfIter   Add performance information every n iterations.
   *                        If zero, do not collect info.
   *  @param[out] perfInfo  Performance information array.
   *
   *  @throws string        if internal linear solve detects singularity,
   *                        or tensor arguments are incompatible.
   */
  template<typename TensorT, typename ExecSpace>
  void cpals_core (const TensorT& x,
                   KtensorT<ExecSpace>& u,
                   const AlgParams& algParams,
                   ttb_indx& numIters,
                   ttb_real& resNorm,
                   const ttb_indx perfIter,
                   PerfHistory& perfInfo,
                   std::ostream& out)
  {
#ifdef HAVE_CALIPER
    cali::Function cali_func("Genten::cpals_core");
#endif

    using std::sqrt;

    const ProcessorMap* pmap = u.getProcessorMap();

    // Whether to use full or symmetric Gram matrix
    const bool full = algParams.full_gram;
    const UploType uplo = Upper;
    bool spd = true; // Use SPD solver if possible

    const ttb_real tol = algParams.tol;
    const ttb_indx maxIters = algParams.maxiters;
    const ttb_real maxSecs = algParams.maxsecs;
    const ttb_indx printIter = algParams.printitn;

    // Check size compatibility of the arguments.
    if (u.isConsistent() == false)
      Genten::error("Genten::cpals_core - ktensor u is not consistent");
    if (x.ndims() != u.ndims())
      Genten::error("Genten::cpals_core - u and x have different num dims");

    // Start timer for total execution time of the algorithm.
    int num_timers = 0;
    const int timer_cpals = num_timers++;
    const int timer_mttkrp = num_timers++;
    //const int timer_mttkrp_local = num_timers++;
    const int timer_comm = num_timers++;
    const int timer_update = num_timers++;
    const int timer_ip = num_timers++;
    const int timer_gramian = num_timers++;
    const int timer_solve = num_timers++;
    const int timer_scale = num_timers++;
    const int timer_norm = num_timers++;
    const int timer_arrange = num_timers++;
    Genten::SystemTimer timer(8, algParams.timings, pmap);

    timer.start(timer_cpals);

    ttb_indx nc = u.ncomponents();     // number of components
    ttb_indx nd = x.ndims();           // number of dimensions

    if (printIter > 0) {
      out << std::endl
             << "CP-ALS:" << std::endl;
      out << "  CP Rank: " << nc << std::endl
          << "  MTTKRP method: "
          << Genten::MTTKRP_Method::names[algParams.mttkrp_method] << std::endl
          << "  Gram formulation: ";
      if (algParams.full_gram)
        out << "full";
      else
        out << "symmetric";
      out << std::endl << std::endl;
    }

    // Distribute the initial guess to have weights of one.
    u.distribute(0);
    Genten::ArrayT<ExecSpace> lambda(nc, (ttb_real) 1.0);

    DistKtensorUpdate<ExecSpace> *dku = createKtensorUpdate(x, u, algParams);
    KtensorT<ExecSpace> u_overlap = dku->createOverlapKtensor(u);
    dku->doImport(u_overlap, u, timer, timer_comm);

    for (ttb_indx  i = 0; i < x.ndims(); i++)
    {
      if (x.size(i) != u_overlap[i].nRows())
        Genten::error("Genten::cpals_core - u and x have different size");
    }

    // Define gamma, an array of Gramian Matrices corresponding to the
    // factor matrices in u.
    // Note that we can skip the first entry of gamma since
    // it's not used in the first inner iteration.
    Genten::FacMatArrayT<ExecSpace> gamma(nd);
    for (ttb_indx n = 0; n < nd; n ++)
    {
      gamma.set_factor( n, FacMatrixT<ExecSpace>(u[n].nCols(), u[n].nCols()) );
    }
    for (ttb_indx n = 1; n < nd; n ++)
    {
      gamma[n].gramian(u[n], full, uplo);
    }

    // Define upsilon to store Hadamard products of the gamma matrices.
    // The matrix is called Z in the Matlab Computations paper.
    Genten::FacMatrixT<ExecSpace> upsilon(nc,nc);

    // Define a temporary matrix used in the loop.
    Genten::FacMatrixT<ExecSpace> tmpMat(nc,nc);

    // Matrix to store the result of MTTKRP for the last mode
    // (Used to compute <x,u> using the trick described by Smith & Karypis)
    Genten::FacMatrixT<ExecSpace> un(u[nd-1].nRows(), nc);
    if (pmap != nullptr)
      un.setProcessorMap(pmap->facMap(nd-1));

    // Pre-calculate the Frobenius norm of the tensor x.
    ttb_real xNorm = x.global_norm();

    ttb_real fit = 0.0;
    ttb_real fitold = 0.0;
    // Compute residual norm and fit of the initial guess.
    // Fit can be a huge negative number for bad start points,
    // so bound it at zero.
    if (perfIter > 0) {
      perfInfo.addEmpty();
      ttb_real dUnorm = sqrt(u.normFsq());
      ttb_real dXtU = innerprod (x, u_overlap, lambda);
      ttb_real dRes = computeResNorm(xNorm, dUnorm, dXtU);
      fit = 1.0 - (dRes / xNorm);
      if (fit < 0.0)
        fit = 0.0;
      auto& p = perfInfo.lastEntry();
      p.iteration = 0;
      p.residual = dRes;
      p.fit = fit;
      p.cum_time = timer.getTotalTime(timer_cpals);
    }

    //--------------------------------------------------
    // Main algorithm loop.
    //--------------------------------------------------
    for (numIters = 0; numIters < maxIters; numIters++)
    {
      fitold = fit;

      // Iterate over all N modes of the tensor
      for (ttb_indx n = 0; n < nd; n++)
      {
        // Update u[n] via MTTKRP with x (Khattri-Rao product).
        // The size of u[n] is dim(n) rows by R columns.
        timer.start(timer_mttkrp);
        Genten::mttkrp (x, u_overlap, n, algParams);
        dku->doExport(u, u_overlap, n, timer, timer_comm, timer_update);
        timer.stop(timer_mttkrp);

        // Save result of MTTKRP for the last mode for computing <x,u>
        if (n == nd-1)
          deep_copy(un, u[n]);

        // Compute the matrix of coefficients in the solve step.
        upsilon = 1;
        for (ttb_indx idx = 0; idx < nd; idx ++)
        {
          if (idx != n)
          {
            upsilon.times(gamma[idx]);
          }
        }

        // Solve upsilon * X = u[n]' for X, and overwrite u[n]
        // with the result.  Equivalent to the Matlab operation
        //   u[n] = (upsilon \ u[n]')'.
        timer.start(timer_solve);
        if (algParams.penalty != ttb_real(0.0))
          upsilon.diagonalShift(algParams.penalty);
        spd = u[n].solveTransposeRHS (upsilon, full, uplo, spd, algParams);
        if (algParams.penalty != ttb_real(0.0))
          upsilon.diagonalShift(-algParams.penalty);
        timer.stop(timer_solve);

        // Compute lambda.
        timer.start(timer_norm);
        if (numIters == 0)
        {
          // L2 norm on first iteration.
          u[n].colNorms(NormTwo, lambda, 0.0);
        }
        else
        {
          // L0 norm (max) on other iterations.
          u[n].colNorms(NormInf, lambda, 1.0);
        }
        timer.stop(timer_norm);

        // Scale u[n] by the inverse of lambda.
        //TBD...this can throw an exception, divide-by-zero,
        //      if a column's norm is zero
        //      I caused it with an unfortunate initial ktensor guess
        timer.start(timer_scale);
        u[n].colScale(lambda, true);
        timer.stop(timer_scale);

        // Update u[n]'s corresponding Gramian matrix.
        timer.start(timer_gramian);
        gamma[n].gramian(u[n], full, uplo);
        timer.stop(timer_gramian);

        dku->doImport(u_overlap, u, n, timer, timer_comm);
      }

      // Compute Frobenius norm of "p", the current factorization
      // consisting of lambda and u.
      upsilon.times(gamma[nd-1]);
      tmpMat.oprod(lambda);
      upsilon.times(tmpMat);
      ttb_real pNorm = sqrt(fabs(upsilon.sum(uplo)));

      // Compute inner product of input data x with "p" using the identity
      // <x,u> = <un,u[nd-1]> where un = mttkrp(x,u,nd-1)
      timer.start(timer_ip);
      //ttb_real xpip = innerprod (x, u, lambda);
      ttb_real xpip = un.innerprod(u[nd-1], lambda);
      timer.stop(timer_ip);

      // Compute the Frobenius norm of the residual using quantities
      // formed earlier.
      resNorm = computeResNorm(xNorm, pNorm, xpip);

      // Compute the relative fit and change since the last iteration.
      fit = 1 - (resNorm / xNorm);
      ttb_real fitchange = fabs(fitold - fit);

      // Print progress of the current iteration.
      if ((printIter > 0) && (((numIters + 1) % printIter) == 0))
      {
        // printf ("Iter %2d: fit = %13.6e  fitdelta = %8.1e\n",
        //         (int) (numIters + 1), fit, fitchange);
        out << "Iter " << std::setw(3) << numIters + 1 << ": fit = "
            << std::setw(13) << std::setprecision(6) << std::scientific << fit
            << " fitdelta = "
            << std::setw(8) << std::setprecision(1) << std::scientific
            << fitchange << std::endl;
      }

      // Fill in performance information if requested.
      if (perfIter > 0 && ((numIters + 1) % perfIter == 0))
      {
        perfInfo.addEmpty();
        auto& p = perfInfo.lastEntry();
        p.iteration = numIters + 1;
        p.residual = resNorm;
        p.fit = fit;
        p.cum_time = timer.getTotalTime(timer_cpals);
      }

      // Check for convergence.
      if ( ((numIters > 0) && (fitchange < tol)) ||
           ((maxSecs >= 0.0) && (timer.getTotalTime(timer_cpals) > maxSecs)) )
      {
        break;
      }
    }
    if (printIter > 0)
      out << "Final fit = " << std::setw(13) << std::setprecision(6)
          << std::scientific << fit << std::endl;

    // Increment so the count starts from one.
    numIters++;

    // Normalize the final result, incorporating the final lambda values.
    u.normalize(Genten::NormTwo);
    lambda.times(u.weights());
    u.setWeights(lambda);
    timer.start(timer_arrange);
    u.arrange();
    timer.stop(timer_arrange);

    timer.stop(timer_cpals);

    // Compute MTTKRP floating-point throughput
    const ttb_real mttkrp_total_time = timer.getTotalTime(timer_mttkrp);
    const ttb_real mttkrp_avg_time = timer.getAvgTime(timer_mttkrp);

    // Use double for these to ensure sufficient precision
    const double atomic = 1.0; // cost of atomic measured in flops
    const double mttkrp_flops =
      x.global_nnz()*nc*(nd+atomic);
    const double mttkrp_reads =
      x.global_nnz()*((nd*nc+3)*sizeof(ttb_real)+nd*sizeof(ttb_indx));
    const double mttkrp_tput =
      ( mttkrp_flops / mttkrp_avg_time ) / (1024.0 * 1024.0 * 1024.0);
    const double mttkrp_factor = mttkrp_flops / mttkrp_reads;

    // Fill in performance information if requested.
    if (perfIter > 0)
    {
      perfInfo.addEmpty();
      auto& p = perfInfo.lastEntry();
      p.iteration = numIters;
      p.residual = resNorm;
      p.fit = fit;
      p.cum_time = timer.getTotalTime(timer_cpals);
      p.mttkrp_throughput = mttkrp_tput;
    }

    if (printIter > 0 && algParams.timings)
    {
      out.setf(std::ios_base::scientific);
      out.precision(2);
      out << "\nCP-ALS completed " << numIters << " iterations in "
          << timer.getTotalTime(timer_cpals) << " seconds\n";
      out << "\tMTTKRP total time = " << mttkrp_total_time
          << " seconds, average time = " << mttkrp_avg_time << " seconds\n";
      out << "\tMTTKRP throughput = " << mttkrp_tput
          << " GFLOP/s, bandwidth factor = " << mttkrp_factor << "\n";
      out << "\tCommunication total time = " << timer.getTotalTime(timer_comm)
          << " seconds, average time = " << timer.getAvgTime(timer_comm)
          << " seconds\n";
      out << "\tInner product total time = " << timer.getTotalTime(timer_ip)
          << " seconds, average time = " << timer.getAvgTime(timer_ip)
          << " seconds\n";
      out << "\tGramian total time = " << timer.getTotalTime(timer_gramian)
          << " seconds, average time = " << timer.getAvgTime(timer_gramian)
          << " seconds\n";
      out << "\tSolve total time = " << timer.getTotalTime(timer_solve)
          << " seconds, average time = " << timer.getAvgTime(timer_solve)
          << " seconds\n";
      out << "\tScale total time = " << timer.getTotalTime(timer_scale)
          << " seconds, average time = " << timer.getAvgTime(timer_scale)
          << " seconds\n";
      out << "\tNorm total time = " << timer.getTotalTime(timer_norm)
          << " seconds, average time = " << timer.getAvgTime(timer_norm)
          << " seconds\n";
      out << "\tArrange total time = " << timer.getTotalTime(timer_arrange)
          << " seconds, average time = " << timer.getAvgTime(timer_arrange)
          << " seconds\n";
    }
    out << std::endl;

    delete dku;

    return;
  }

  //------------------------------------------------------
  // PRIVATE FUNCTIONS
  //------------------------------------------------------

  /*
   * Compute the residual Frobenius norm between data tensor X
   * and Ktensor model M as
   *     sqrt{ |X|^2_F - 2(X . M) + |M|^2_F }
   *
   * The residual can be slightly negative due to roundoff errors
   * if the model is a nearly exact fit to the data.  The threshold
   * for fatal error was determined from experimental observations.
   */
  static ttb_real computeResNorm(const ttb_real  xNorm,
                                 const ttb_real  mNorm,
                                 const ttb_real  xDotm)
  {
    using std::sqrt;

    ttb_real  d = (xNorm * xNorm) + (mNorm * mNorm) - (2 * xDotm);

    ttb_real  result;
    ttb_real dSmallNegThresh = -(xDotm * sqrt(MACHINE_EPSILON) * 1e3);
    if (d > DBL_MIN)
      result = sqrt(d);
    else if (d > dSmallNegThresh)
      result = 0.0;
    else
    {
      // Warn the user that the residual norm is negative instead of throwing
      // an error.
      std::ostringstream  sMsg;
      sMsg.setf(std::ios_base::scientific);
      sMsg.precision(15);
      sMsg << "Genten::cpals_core - residual norm^2, " << d
           << ", is negative:" << std::endl
           << "\t||X||^2 = " << xNorm*xNorm << "," << std::endl
           << "\t||M||^2 = " << mNorm*mNorm << "," << std::endl
           << "\t<X,M>   = " << xDotm << "." << std::endl
           << "This likely means the gram matrix is (nearly) singular.\n"
           << "Try adding regularization by making the penalty term nonzero\n"
           << "(e.g., --penalty 1e-6) or using the rank-deficient "
           << "least-squares solver (LAPACK's GELSY)\n"
           << "(e.g., --full-gram --rank-deficient-solver --rcond 1e-8)."
           << std::endl;
      Genten::error(sMsg.str());
      //std::cout << sMsg.str() << std::endl;
      result = 0.0;
    }
    return( result );
  }

}

#define INST_MACRO(SPACE)                                               \
  template void cpals_core<SptensorT<SPACE>,SPACE>(                     \
    const SptensorT<SPACE>& x,                                          \
    KtensorT<SPACE>& u,                                                 \
    const AlgParams& algParams,                                         \
    ttb_indx& numIters,                                                 \
    ttb_real& resNorm,                                                  \
    const ttb_indx perfIter,                                            \
    PerfHistory& perfInfo,                                              \
    std::ostream& out);                                                 \
                                                                        \
  template void cpals_core<TensorT<SPACE>,SPACE>(                       \
    const TensorT<SPACE>& x,                                            \
    KtensorT<SPACE>& u,                                                 \
    const AlgParams& algParams,                                         \
    ttb_indx& numIters,                                                 \
    ttb_real& resNorm,                                                  \
    const ttb_indx perfIter,                                            \
    PerfHistory& perfInfo,                                              \
    std::ostream& out);

GENTEN_INST(INST_MACRO)
