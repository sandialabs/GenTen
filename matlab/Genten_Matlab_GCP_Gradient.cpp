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

// Mex Driver for computing permutation arrays

#include "Genten_Matlab.hpp"
#include "Genten_GCP_SamplingKernels.hpp"
#include "Genten_GCP_LossFunctions.hpp"
#include "Genten_GCP_SS_Grad.hpp"
#include "Genten_GCP_Sampler.hpp"
#include "Genten_MixedFormatOps.hpp"
#include "Genten_SystemTimer.hpp"

#include "Kokkos_Random.hpp"

template <typename ExecSpace, typename LossFunction>
Genten::KtensorT<ExecSpace>
gcp_gradient_driver(
  const std::string& method,
  Genten::SptensorT<ExecSpace>& X,
  const ttb_indx num_samples_nonzeros,
  const ttb_indx num_samples_zeros,
  const ttb_real weight_nonzeros,
  const ttb_real weight_zeros,
  const Genten::KtensorT<ExecSpace>& u,
  const Genten::KtensorT<ExecSpace>& uprev,
  const Genten::ArrayT<ExecSpace>& window,
  const ttb_real window_penalty,
  const LossFunction& loss_func,
  const Genten::IndxArrayT<ExecSpace> modes,
  const ttb_real penalty,
  Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
  const Genten::AlgParams& algParams)
{
  // to do:  call mttkrp-all

  const ttb_indx d = modes.size();
  const ttb_indx nd = u.ndims();
  const ttb_indx nc = u.ncomponents();

  Genten::IndxArray modes_host = create_mirror_view(
    typename Genten::IndxArray::exec_space(), modes);
  deep_copy(modes_host, modes);

  // Initialize gradient
  Genten::KtensorT<ExecSpace> G(nc,d); // length(modes), not u.ndims()
  for (ttb_indx k=0; k<d; ++k) {
    const ttb_indx kk = modes_host[k];
    Genten::FacMatrixT<ExecSpace> v(u[kk].nRows(), nc); // initializes to zero
    G.set_factor(k, v);
  }

  // Initialize ut == u with time mode replaced by uprev
  const bool have_uprev = (uprev.ncomponents() > 0 && uprev.ndims() > 0);
  Genten::KtensorT<ExecSpace> ut;
  if (have_uprev) {
    ut = Genten::KtensorT<ExecSpace>(nc, nd);
    for (ttb_indx i=0; i<nd-1; ++i) {
      Genten::FacMatrixT<ExecSpace> v(u[i].nRows(), nc);
      deep_copy(v, u[i]);
      ut.set_factor(i, v);
    }
    Genten::FacMatrixT<ExecSpace> v(uprev[nd-1].nRows(), nc);
    deep_copy(v, uprev[nd-1]);
    ut.set_factor(nd-1,v);
    ut.setWeights(1.0);
    uprev.setWeights(1.0);
  }

  if (method == "semi-stratified" && algParams.fuse) {
    Genten::SystemTimer timer(2, algParams.timings);
    Genten::Impl::gcp_sgd_ss_grad(
      X, u, ut, uprev, loss_func,
      num_samples_nonzeros, num_samples_zeros,
      weight_nonzeros, weight_zeros,
      window, window_penalty,
      modes, G, rand_pool, algParams, timer, 0, 1);
  }
  else {
    // Compute sampled gradient tensor
    Genten::SptensorT<ExecSpace> Y;
    Genten::ArrayT<ExecSpace> w;
    if (method == "stratified") {
      if (algParams.hash) {
        auto hash = Genten::Sampler<Genten::SptensorT<ExecSpace>,LossFunction>::buildHashMap(X, std::cout);
        Genten::Impl::stratified_sample_tensor(
          X, Genten::Impl::HashSearcher<ExecSpace>(X.impl(), hash),
          num_samples_nonzeros, num_samples_zeros,
          weight_nonzeros, weight_zeros,
          u, Genten::Impl::StratifiedGradient<LossFunction>(loss_func), true,
          Y, w, rand_pool, algParams);
      }
      else {
        if (!X.isSorted())
          X.sort();
        Genten::Impl::stratified_sample_tensor(
          X, Genten::Impl::SortSearcher<ExecSpace>(X.impl()),
          num_samples_nonzeros, num_samples_zeros,
          weight_nonzeros, weight_zeros,
          u, Genten::Impl::StratifiedGradient<LossFunction>(loss_func), true,
          Y, w, rand_pool, algParams);
      }
    }
    else if (method == "semi-stratified")
      Genten::Impl::stratified_sample_tensor(
        X, Genten::Impl::SemiStratifiedSearcher<ExecSpace>(),
        num_samples_nonzeros, num_samples_zeros,
        weight_nonzeros, weight_zeros,
        u, Genten::Impl::SemiStratifiedGradient<LossFunction>(loss_func), true,
        Y, w, rand_pool, algParams);
    else {
      std::string err = "Unknown method " + method;
      throw err;
    }

    // Compute gradient
    if (algParams.mttkrp_method == Genten::MTTKRP_Method::Perm &&
        !Y.havePerm())
      Y.createPermutation();
    for (ttb_indx k=0; k<d; ++k) {
      const ttb_indx kk = modes_host[k];
      Genten::mttkrp(Y, u, kk, G[k], algParams);
    }

    // Add uprev term
    if (have_uprev) {
      Genten::SptensorT<ExecSpace> Yt;

      Genten::Impl::stratified_ktensor_grad(
        Y, num_samples_nonzeros, num_samples_zeros,
        weight_nonzeros, weight_zeros,
        ut, uprev, window, window_penalty, loss_func,
        Yt, algParams);

      // const ttb_indx num_samples = num_samples_nonzeros+num_samples_zeros;
      // const ttb_real weight = X.numel_float() / ttb_real(num_samples);
      // Genten::Impl::stratified_ktensor_grad(
      //   Y, num_samples, ttb_indx(0),
      //   weight, ttb_real(0.0),
      //   ut, uprev, window, window_penalty, loss_func,
      //   Yt, algParams);

      // Genten::Impl::uniform_ktensor_grad(
      //   num_samples, weight, ut, uprev, window, window_penalty, loss_func,
      //   Yt, rand_pool, algParams);

      if (algParams.mttkrp_method == Genten::MTTKRP_Method::Perm &&
          !Yt.havePerm())
        Yt.createPermutation();
      for (ttb_indx k=0; k<d; ++k) {
        const ttb_indx kk = modes_host[k];
        Genten::FacMatrixT<ExecSpace> v(u[kk].nRows(), nc);
        Genten::mttkrp(Yt, ut, kk, v, algParams);
        G[k].plus(v);
      }
    }
  }

  // to do:  saxpy
  if (penalty != 0.0) {
    for (ttb_indx k=0; k<d; ++k) {
      const ttb_indx kk = modes_host[k];
      Genten::FacMatrixT<ExecSpace> v(u[kk].nRows(), nc);
      deep_copy(v, u[kk]);
      v.times(penalty);
      G[k].plus(v);
    }
  }

  return G;
}

extern "C" {

DLL_EXPORT_SYM void mexFunction(int nlhs, mxArray *plhs[],
                                int nrhs, const mxArray *prhs[])
{
  typedef Genten::DefaultExecutionSpace ExecSpace;
  typedef Genten::SptensorT<ExecSpace> Sptensor_type;
  typedef Genten::KtensorT<ExecSpace> Ktensor_type;
  typedef Genten::IndxArrayT<ExecSpace> indx_array_type;
  typedef Genten::ArrayT<ExecSpace> array_type;

  GentenInitialize();

  try {
    if (nrhs < 13 || nlhs != 1) {
      std::string err = "Expected at least 13 input and 1 output arguments";
      throw err;
    }

    Genten::AlgParams algParams;
    algParams.fixup<ExecSpace>(std::cout);

    // Parse inputs
    int arg = 0;
    const std::string method = mxGetStdString(prhs[arg++]);
    Sptensor_type X = mxGetSptensor<ExecSpace>(prhs[arg++]);
    const ttb_indx num_samples_nonzeros =
      static_cast<ttb_indx>(mxGetScalar(prhs[arg++]));
    const ttb_indx num_samples_zeros =
      static_cast<ttb_indx>(mxGetScalar(prhs[arg++]));
    const ttb_real weight_nonzeros =
      static_cast<ttb_real>(mxGetScalar(prhs[arg++]));
    const ttb_real weight_zeros =
      static_cast<ttb_real>(mxGetScalar(prhs[arg++]));
    const Ktensor_type u = mxGetKtensor<ExecSpace>(prhs[arg++]);
    const Ktensor_type uprev = mxGetKtensor<ExecSpace>(prhs[arg++]);
    const array_type window = mxGetArray<ExecSpace>(prhs[arg++]);
    const ttb_real window_penalty = mxGetScalar(prhs[arg++]);
    const std::string loss_type = mxGetStdString(prhs[arg++]);
    const indx_array_type modes = mxGetIndxArray<ExecSpace>(prhs[arg++],true);
    const ttb_real penalty =
      static_cast<ttb_real>(mxGetScalar(prhs[arg++]));
    if (nrhs >= arg+1) {
      auto args = mxBuildArgList(nrhs, arg, prhs);
      algParams.parse(args);
      if (Genten::check_and_print_unused_args(args, std::cout)) {
        algParams.print_help(std::cout);
        std::string err = "Invalid command line arguments.";
        throw err;
      }
    }
    algParams.loss_function_type = loss_type;

    // to do:  figure out how to reuse this across calls, as it is expensive
    // to recreate each time
    Kokkos::Random_XorShift64_Pool<ExecSpace> rand_pool(rand());

    // Dispatch implementation based on loss function type
    Ktensor_type G;
    Genten::dispatch_loss(algParams, [&](const auto& loss)
    {
      G =
        gcp_gradient_driver(method,X,num_samples_nonzeros,num_samples_zeros,
                            weight_nonzeros,weight_zeros,u,uprev,
                            window,window_penalty, loss,
                            modes,penalty,rand_pool,algParams);
    });

    // Set output
    plhs[0] = mxSetKtensor(G);
  }

  catch(std::string sExc)
  {
    std::cout << "*** Call to genten threw an exception:\n";
    std::cout << "  " << sExc << "\n";
  }
}

}
