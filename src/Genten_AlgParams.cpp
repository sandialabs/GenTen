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

#include "Genten_AlgParams.hpp"
#include "Genten_FacMatrix.hpp"

Genten::AlgParams::AlgParams() :
  exec_space(Execution_Space::default_type),
  method(Solver_Method::default_type),
  rank(16),
  seed(12345),
  prng(false),
  maxiters(100),
  maxsecs(-1.0),
  tol(0.0004),
  printitn(1),
  debug(false),
  timings(false),
  full_gram(FacMatrixT<DefaultExecutionSpace>::full_gram_default),
  rank_def_solver(false),
  rcond(1e-8),
  penalty(0.0),
  dist_guess_method("serial"),
  mttkrp_method(MTTKRP_Method::default_type),
  mttkrp_all_method(MTTKRP_All_Method::default_type),
  mttkrp_nnz_tile_size(128),
  mttkrp_duplicated_factor_matrix_tile_size(0),
  mttkrp_duplicated_threshold(-1.0),
  warmup(false),
  ttm_method(TTM_Method::default_type),
  opt_method(Opt_Method::default_type),
  lower(-DOUBLE_MAX),
  upper(DOUBLE_MAX),
  rolfilename(""),
  factr(1e7),
  pgtol(1e-5),
  memory(5),
  max_total_iters(5000),
  hess_vec_method(Hess_Vec_Method::default_type),
  loss_function_type(Genten::GCP_LossFunction::default_type),
  loss_eps(1.0e-10),
  gcp_tol(-DOUBLE_MAX),
  sampling_type(Genten::GCP_Sampling::default_type),
  rate(1.0e-3),
  decay(0.1),
  max_fails(10),
  epoch_iters(1000),
  frozen_iters(1),
  rng_iters(128),
  num_samples_nonzeros_value(0),
  num_samples_zeros_value(0),
  num_samples_nonzeros_grad(0),
  num_samples_zeros_grad(0),
  oversample_factor(1.1),
  bulk_factor(10),
  w_f_nz(-1.0), // Means to compute the default
  w_f_z(-1.0),
  w_g_nz(-1.0),
  w_g_z(-1.0),
  hash(false),
  fuse(false),
  fuse_sa(false),
  compute_fit(false),
  step_type(Genten::GCP_Step::ADAM),
  adam_beta1(0.9),    // Defaults taken from ADAM paper
  adam_beta2(0.999),
  adam_eps(1.0e-8),
  async(false),
  anneal(false),
  anneal_min_lr(1e-14),
  anneal_max_lr(1e-10)
{}

void Genten::AlgParams::parse(std::vector<std::string>& args)
{
  // Parse options from command-line, using default values set above as defaults

  // Generic options
  exec_space = parse_ttb_enum(args, "--exec-space", exec_space,
                              Genten::Execution_Space::num_types,
                              Genten::Execution_Space::types,
                              Genten::Execution_Space::names);
  method = parse_ttb_enum(args, "--method", method,
                          Genten::Solver_Method::num_types,
                          Genten::Solver_Method::types,
                          Genten::Solver_Method::names);
  rank = parse_ttb_indx(args, "--rank", rank, 1, INT_MAX);
  seed = parse_ttb_indx(args, "--seed", seed, 0, INT_MAX);
  prng = parse_ttb_bool(args, "--prng", "--no-prng", prng);
  maxiters = parse_ttb_indx(args, "--maxiters", maxiters, 1, INT_MAX);
  maxsecs = parse_ttb_real(args, "--maxsecs", maxsecs, -1.0, DOUBLE_MAX);
  tol = parse_ttb_real(args, "--tol", tol, 0.0, DOUBLE_MAX);
  printitn = parse_ttb_indx(args, "--printitn", printitn, 0, INT_MAX);
  debug = parse_ttb_bool(args, "--debug", "--no-debug", debug);
  timings = parse_ttb_bool(args, "--timings", "--no-timings", timings);
  full_gram = parse_ttb_bool(args, "--full-gram", "--sym-gram", full_gram);
  rank_def_solver = parse_ttb_bool(args, "--rank-def-solver",
                                   "--no-rank-def-solver", rank_def_solver);
  rcond = parse_ttb_real(args, "--rcond", rcond, 0.0, DOUBLE_MAX);
  penalty = parse_ttb_real(args, "--penalty", penalty, 0.0, DOUBLE_MAX);
  dist_guess_method = parse_string(args, "--dist-guess", dist_guess_method);

  // MTTKRP options
  mttkrp_method = parse_ttb_enum(args, "--mttkrp-method", mttkrp_method,
                                 Genten::MTTKRP_Method::num_types,
                                 Genten::MTTKRP_Method::types,
                                 Genten::MTTKRP_Method::names);
  mttkrp_all_method = parse_ttb_enum(args, "--mttkrp-all-method",
                                     mttkrp_all_method,
                                     Genten::MTTKRP_All_Method::num_types,
                                     Genten::MTTKRP_All_Method::types,
                                     Genten::MTTKRP_All_Method::names);
  mttkrp_nnz_tile_size =
    parse_ttb_indx(args, "--mttkrp-nnz-tile-size",
                   mttkrp_nnz_tile_size, 1, INT_MAX);
  mttkrp_duplicated_factor_matrix_tile_size =
    parse_ttb_indx(args, "--mttkrp-duplicated-tile-size",
                   mttkrp_duplicated_factor_matrix_tile_size, 0, INT_MAX);
   mttkrp_duplicated_threshold =
    parse_ttb_real(args, "--mttkrp-duplicated-threshold",
                   mttkrp_duplicated_factor_matrix_tile_size, -1.0, DOUBLE_MAX);
  warmup = parse_ttb_bool(args, "--warmup", "--no-warmup", warmup);

  // TTM options
  ttm_method = parse_ttb_enum(args, "--ttm-method", ttm_method,
                                 Genten::TTM_Method::num_types,
                                 Genten::TTM_Method::types,
                                 Genten::TTM_Method::names);

  // CP-Opt options
  opt_method = parse_ttb_enum(args, "--opt", opt_method,
                                 Genten::Opt_Method::num_types,
                                 Genten::Opt_Method::types,
                                 Genten::Opt_Method::names);
  lower = parse_ttb_real(args, "--lower", lower, -DOUBLE_MAX, DOUBLE_MAX);
  upper = parse_ttb_real(args, "--upper", upper, -DOUBLE_MAX, DOUBLE_MAX);
  rolfilename = parse_string(args, "--rol", rolfilename.c_str());
  factr = parse_ttb_real(args, "--factr", factr, 0.0, DOUBLE_MAX);
  pgtol = parse_ttb_real(args, "--pgtol", pgtol, 0.0, DOUBLE_MAX);
  memory = parse_ttb_indx(args, "--memory", memory, 0, INT_MAX);
  max_total_iters = parse_ttb_indx(args, "--total-iters", max_total_iters, 0, INT_MAX);
  hess_vec_method = parse_ttb_enum(args, "--hessian", hess_vec_method,
                                 Genten::Hess_Vec_Method::num_types,
                                 Genten::Hess_Vec_Method::types,
                                 Genten::Hess_Vec_Method::names);

  // GCP options
  loss_function_type = parse_ttb_enum(args, "--type", loss_function_type,
                                      Genten::GCP_LossFunction::num_types,
                                      Genten::GCP_LossFunction::types,
                                      Genten::GCP_LossFunction::names);
  loss_eps = parse_ttb_real(args, "--eps", loss_eps, 0.0, 1.0);
  gcp_tol = parse_ttb_real(args, "--gcp-tol", gcp_tol, -DOUBLE_MAX, DOUBLE_MAX);

  // GCP-SGD options
  sampling_type = parse_ttb_enum(args, "--sampling",
                                 sampling_type,
                                 Genten::GCP_Sampling::num_types,
                                 Genten::GCP_Sampling::types,
                                 Genten::GCP_Sampling::names);
  rate = parse_ttb_real(args, "--rate", rate, 0.0, DOUBLE_MAX);
  decay = parse_ttb_real(args, "--decay", decay, 0.0, 1.0);
  max_fails = parse_ttb_indx(args, "--fails", max_fails, 0, INT_MAX);
  epoch_iters =
    parse_ttb_indx(args, "--epochiters", epoch_iters, 1, INT_MAX);
  frozen_iters =
    parse_ttb_indx(args, "--frozeniters", frozen_iters, 1, INT_MAX);
  rng_iters =
    parse_ttb_indx(args, "--rngiters", rng_iters, 1, INT_MAX);
  num_samples_nonzeros_value =
    parse_ttb_indx(args, "--fnzs", num_samples_nonzeros_value, 0, INT_MAX);
  num_samples_zeros_value =
    parse_ttb_indx(args, "--fzs", num_samples_zeros_value, 0, INT_MAX);
  num_samples_nonzeros_grad =
    parse_ttb_indx(args, "--gnzs", num_samples_nonzeros_grad, 0, INT_MAX);
  num_samples_zeros_grad =
    parse_ttb_indx(args, "--gzs", num_samples_zeros_grad, 0, INT_MAX);
  oversample_factor = parse_ttb_real(args, "--oversample",
                                     oversample_factor, 1.0, DOUBLE_MAX);
  bulk_factor =
    parse_ttb_indx(args, "--bulk-factor", bulk_factor, 1, INT_MAX);
  w_f_nz = parse_ttb_real(args, "--fnzw", w_f_nz, -1.0, DOUBLE_MAX);
  w_f_z = parse_ttb_real(args, "--fzw", w_f_z, -1.0, DOUBLE_MAX);
  w_g_nz = parse_ttb_real(args, "--gnzw", w_g_nz, -1.0, DOUBLE_MAX);
  w_g_z = parse_ttb_real(args, "--gzw", w_g_z, -1.0, DOUBLE_MAX);
  hash = parse_ttb_bool(args, "--hash", "--no-hash", hash);
  fuse = parse_ttb_bool(args, "--fuse", "--no-fuse", fuse);
  fuse_sa = parse_ttb_bool(args, "--fuse-sa", "--no-fuse-sa", fuse_sa);
  compute_fit = parse_ttb_bool(args, "--fit", "--no-fit", compute_fit);
  step_type = parse_ttb_enum(args, "--step", step_type,
                             Genten::GCP_Step::num_types,
                             Genten::GCP_Step::types,
                             Genten::GCP_Step::names);
  adam_beta1 = parse_ttb_real(args, "--adam-beta1", adam_beta1, 0.0, 1.0);
  adam_beta2 = parse_ttb_real(args, "--adam-beta2", adam_beta2, 0.0, 1.0);
  adam_eps = parse_ttb_real(args, "--adam-eps", adam_eps, 0.0, 1.0);
  async = parse_ttb_bool(args, "--async", "--sync", async);
  anneal = parse_ttb_bool(args, "--anneal", "--no-anneal", anneal);
  anneal_min_lr = parse_ttb_real(args, "--anneal-min-lr", anneal_min_lr, 0.0, 1.0);
  anneal_max_lr = parse_ttb_real(args, "--anneal-max-lr", anneal_max_lr, 0.0, 1.0);
}

void Genten::AlgParams::parse(const ptree& input)
{
  // Generic options
  parse_ptree_enum<Execution_Space>(input, "exec-space", exec_space);

  // mttkrp tree may appear in multiple places, so make a lambda to parse it
  auto parse_mttkrp = [&](const ptree& tree) {
    auto mttkrp_input_o = tree.get_child_optional("mttkrp");
    if (mttkrp_input_o) {
      auto& mttkrp_input = *mttkrp_input_o;
      parse_ptree_enum<MTTKRP_Method>(mttkrp_input, "method", mttkrp_method);
      parse_ptree_enum<MTTKRP_All_Method>(mttkrp_input, "all-method", mttkrp_all_method);
      parse_ptree_value(mttkrp_input, "nnz-tile-size", mttkrp_nnz_tile_size,
                        1, INT_MAX);
      parse_ptree_value(mttkrp_input, "duplicated-tile-size",
                        mttkrp_duplicated_factor_matrix_tile_size, 0, INT_MAX);
      parse_ptree_value(mttkrp_input, "duplicated-threshold",
                        mttkrp_duplicated_threshold, -1.0, DOUBLE_MAX);
      parse_ptree_value(mttkrp_input, "warmup", warmup);
    }
  };

  // ktensor
  auto ktensor_input_o = input.get_child_optional("k-tensor");
  if (ktensor_input_o) {
    auto& ktensor_input = *ktensor_input_o;
    parse_ptree_value(ktensor_input, "rank", rank, 1, INT_MAX);
    parse_ptree_value(ktensor_input, "seed", seed, 0, INT_MAX);
    parse_ptree_value(ktensor_input, "prng", prng);
    parse_ptree_value(ktensor_input, "distributed-guess", dist_guess_method);
  }

  // CP-ALS
  auto cpals_input_o = input.get_child_optional("cp-als");
  if (cpals_input_o) {
    auto& cpals_input = *cpals_input_o;
    parse_ptree_value(cpals_input, "maxiters", maxiters, 1, INT_MAX);
    parse_ptree_value(cpals_input, "maxsecs", maxsecs, -1.0, DOUBLE_MAX);
    parse_ptree_value(cpals_input, "tol", tol, 0.0, DOUBLE_MAX);
    parse_ptree_value(cpals_input, "printitn", printitn, 0, INT_MAX);
    parse_ptree_value(cpals_input, "debug", debug);
    parse_ptree_value(cpals_input, "timings", timings);
    parse_ptree_value(cpals_input, "full-gram", full_gram);
    parse_ptree_value(cpals_input, "rank-def-solver", rank_def_solver);
    parse_ptree_value(cpals_input, "rcond", rcond, 0.0, DOUBLE_MAX);
    parse_ptree_value(cpals_input, "penalty", penalty, 0.0, DOUBLE_MAX);
    parse_mttkrp(cpals_input);
  }

  // TTM
  auto ttm_input_o = input.get_child_optional("ttm");
  if (ttm_input_o) {
    auto& ttm_input = *ttm_input_o;
    parse_ptree_enum<TTM_Method>(ttm_input, "method", ttm_method);
  }

  // CP-OPT
  auto cpopt_input_o = input.get_child_optional("cp-opt");
  if (cpopt_input_o) {
    auto& cpopt_input = *cpopt_input_o;
    parse_ptree_enum<Opt_Method>(cpopt_input, "method", opt_method);
    parse_ptree_value(cpopt_input, "lower", lower, -DOUBLE_MAX, DOUBLE_MAX);
    parse_ptree_value(cpopt_input, "upper", upper, -DOUBLE_MAX, DOUBLE_MAX);
    parse_ptree_value(cpopt_input, "rol", rolfilename);
    parse_ptree_value(cpopt_input, "factr", factr, 0.0, DOUBLE_MAX);
    parse_ptree_value(cpopt_input, "pgtol", pgtol, 0.0, DOUBLE_MAX);
    parse_ptree_value(cpopt_input, "memory", memory, 0, INT_MAX);
    parse_ptree_value(cpopt_input, "total-iters", max_total_iters, 0, INT_MAX);
  }

  // GCP
  auto gcp_input_o = input.get_child_optional("gcp");
  if (gcp_input_o) {
    auto& gcp_input = *gcp_input_o;
    parse_ptree_enum<GCP_LossFunction>(gcp_input, "type", loss_function_type);
    parse_ptree_value(gcp_input, "eps", loss_eps, 0.0, 1.0);
    parse_ptree_value(gcp_input, "tol", gcp_tol, -DOUBLE_MAX, DOUBLE_MAX);

    // GCP-SGD
    parse_ptree_enum<GCP_Sampling>(gcp_input, "sampling", sampling_type);
    parse_ptree_value(gcp_input, "rate", rate, 0.0, DOUBLE_MAX);
    parse_ptree_value(gcp_input, "decay", decay, 0.0, 1.0);
    parse_ptree_value(gcp_input, "fails", max_fails, 0, INT_MAX);
    parse_ptree_value(gcp_input, "epochiters", epoch_iters, 1, INT_MAX);
    parse_ptree_value(gcp_input, "frozeniters", frozen_iters, 1, INT_MAX);
    parse_ptree_value(gcp_input, "rngiters", rng_iters, 1, INT_MAX);
    parse_ptree_value(gcp_input, "fnzs", num_samples_nonzeros_value, 0, INT_MAX);
    parse_ptree_value(gcp_input, "fzs", num_samples_zeros_value, 0, INT_MAX);
    parse_ptree_value(gcp_input, "gnzs", num_samples_nonzeros_grad, 0, INT_MAX);
    parse_ptree_value(gcp_input, "gzs", num_samples_zeros_grad, 0, INT_MAX);
    parse_ptree_value(gcp_input, "oversample", oversample_factor, 1.0, DOUBLE_MAX);
    parse_ptree_value(gcp_input, "bulk-factor", bulk_factor, 1, INT_MAX);
    parse_ptree_value(gcp_input, "fnzw", w_f_nz, -1.0, DOUBLE_MAX);
    parse_ptree_value(gcp_input, "fzw", w_f_z, -1.0, DOUBLE_MAX);
    parse_ptree_value(gcp_input, "gnzw", w_g_nz, -1.0, DOUBLE_MAX);
    parse_ptree_value(gcp_input, "gzw", w_g_z, -1.0, DOUBLE_MAX);
    parse_ptree_value(gcp_input, "hash", hash);
    parse_ptree_value(gcp_input, "fuse", fuse);
    parse_ptree_value(gcp_input, "fuse-sa", fuse_sa);
    parse_ptree_value(gcp_input, "fit", compute_fit);
    parse_ptree_enum<GCP_Step>(gcp_input, "step", step_type);
    parse_ptree_value(gcp_input, "adam-beta1", adam_beta1, 0.0, 1.0);
    parse_ptree_value(gcp_input, "adam-beta2", adam_beta2, 0.0, 1.0);
    parse_ptree_value(gcp_input, "adam-eps", adam_eps, 0.0, 1.0);
    parse_ptree_value(gcp_input, "async", async);
    parse_ptree_value(gcp_input, "anneal", anneal);
    parse_ptree_value(gcp_input, "anneal-min-lr", anneal_min_lr, 0.0, 1.0);
    parse_ptree_value(gcp_input, "anneal-max-lr", anneal_max_lr, 0.0, 1.0);
  }
}

void Genten::AlgParams::print_help(std::ostream& out)
{
  out << "Generic options: " << std::endl;
  out << "  --exec-space <space> execution space to run on: ";
  for (unsigned i=0; i<Genten::Execution_Space::num_types; ++i) {
    out << Genten::Execution_Space::names[i];
    if (i != Genten::Execution_Space::num_types-1)
      out << ", ";
  }
  out << std::endl;
  out << "  --method <method>  decomposition method: ";
  for (unsigned i=0; i<Genten::Solver_Method::num_types; ++i) {
    out << Genten::Solver_Method::names[i];
    if (i != Genten::Solver_Method::num_types-1)
      out << ", ";
  }
  out << std::endl;
  out << "  --rank <int>       rank of factorization to compute" << std::endl;
  out << "  --seed <int>       seed for random number generator used in initial guess" << std::endl;
  out << "  --prng             use parallel random number generator (not consistent with Matlab)" << std::endl;
  out << "  --maxiters <int>   maximum iterations to perform" << std::endl;
  out << "  --maxsecs <float>  maximum running time" << std::endl;
  out << "  --tol <float>      stopping tolerance" << std::endl;
  out << "  --printitn <int>   print every <int>th iteration; 0 for no printing" << std::endl;
  out << "  --debug            turn on debugging output" << std::endl;
  out << "  --timings          print accurate kernel timing info (but may increase total run time by adding fences)" << std::endl;
  out << "  --full-gram        use full Gram matrix formulation (which may be faster than the symmetric formulation on some architectures)" << std::endl;
  out << "  --rank-def-solver  use rank-deficient least-squares solver (GELSY) with full-gram formluation (useful when gram matrix is singular)" << std::endl;
  out << "  --rcond <float>    truncation parameter for rank-deficient solver" << std::endl;
  out << "  --penalty <float>  penalty term for regularization (useful if gram matrix is singular)" << std::endl;
  out << "  --dist-guess <string> method for distributed initial guess" << std::endl;

  out << std::endl;
  out << "MTTKRP options:" << std::endl;
  out << "  --mttkrp-method <method> MTTKRP algorithm: ";
  for (unsigned i=0; i<Genten::MTTKRP_Method::num_types; ++i) {
    out << Genten::MTTKRP_Method::names[i];
    if (i != Genten::MTTKRP_Method::num_types-1)
      out << ", ";
  }
  out << std::endl;
  out << "  --mttkrp-all-method <method> MTTKRP algorithm for all modes simultaneously: ";
  for (unsigned i=0; i<Genten::MTTKRP_All_Method::num_types; ++i) {
    out << Genten::MTTKRP_All_Method::names[i];
    if (i != Genten::MTTKRP_All_Method::num_types-1)
      out << ", ";
  }
  out << std::endl;
  out << "  --mttkrp-nnz-tile-size <int> Nonzero tile size for mttkrp algorithm"
      << std::endl;
  out << "  --mttkrp-duplicated-tile-size <int> Factor matrix tile size for duplicated mttkrp algorithm" << std::endl;
  out << "  --mttkrp-duplicated-threshold <float> Theshold for determining when to not use duplicated mttkrp algorithm (set to -1.0 to always use duplicated)" << std::endl;
  out << "  --warmup           do an iteration of mttkrp to warmup (useful for generating accurate timing information)" << std::endl;

  out << std::endl;
  out << "TTM options:" << std::endl;
  out << "  --ttm-method <method> TTM algorithm: ";
  for (unsigned i=0; i<Genten::TTM_Method::num_types; ++i) {
    out << Genten::TTM_Method::names[i];
    if (i != Genten::TTM_Method::num_types-1)
      out << ", ";
  } out << std::endl;
  out << std::endl;
  out << "CP-Opt options:" << std::endl;
  out << "  --opt <method> optimization method: ";
  for (unsigned i=0; i<Genten::Opt_Method::num_types; ++i) {
    out << Genten::Opt_Method::names[i];
    if (i != Genten::Opt_Method::num_types-1)
      out << ", ";
  }
  out << std::endl;
  out << "  --lower <float>    lower bound of factorization" << std::endl;
  out << "  --upper <float>    upper bound of factorization" << std::endl;
  out << "  --rol <string>     path to ROL optimization settings file for CP-Opt method" << std::endl;
  out << "  --factr <float>    factr parameter for L-BFGS-B" << std::endl;
  out << "  --pgtol <float>    pgtol parameter for L-BFGS-B" << std::endl;
  out << "  --memory <int>     memory parameter for L-BFGS-B" << std::endl;
  out << "  --total-iters <int>  max total iterations for L-BFGS-B" << std::endl;
  out << "  --hessian <method> Hessian-vector product method: ";
  for (unsigned i=0; i<Genten::Hess_Vec_Method::num_types; ++i) {
    out << Genten::Hess_Vec_Method::names[i];
    if (i != Genten::Hess_Vec_Method::num_types-1)
      out << ", ";
  }
  out << std::endl;
  out << "GCP options:" << std::endl;
  out << "  --type <type>      loss function type for GCP: ";
  for (unsigned i=0; i<Genten::GCP_LossFunction::num_types; ++i) {
    out << Genten::GCP_LossFunction::names[i];
    if (i != Genten::GCP_LossFunction::num_types-1)
      out << ", ";
  }
  out << std::endl;
  out << "  --eps <float>      perturbation of loss functions for entries near 0" << std::endl;
  out << "  --gcp-tol <float> GCP solver tolerance" << std::endl;

  out << std::endl;
  out << "GCP-SGD options:" << std::endl;
  out << "  --sampling <type> sampling method for GCP-SGD: ";
  for (unsigned i=0; i<Genten::GCP_Sampling::num_types; ++i) {
    out << Genten::GCP_Sampling::names[i];
    if (i != Genten::GCP_Sampling::num_types-1)
      out << ", ";
  }
  out << std::endl;
  out << "  --rate <float>     initial step size" << std::endl;
  out << "  --decay <float>    rate step size decreases on fails" << std::endl;
  out << "  --fails <int>      maximum number of fails" << std::endl;
  out << "  --epochiters <int> iterations per epoch" << std::endl;
  out << "  --frozeniters <int> inner iterations with frozen gradient"
      << std::endl;
  out << "  --rngiters <int>   iteration loops in parallel RNG" << std::endl;
  out << "  --fnzs <int>       nonzero samples for f-est" << std::endl;
  out << "  --fzs <int>        zero samples for f-est" << std::endl;
  out << "  --gnzs <int>       nonzero samples for gradient" << std::endl;
  out << "  --gzs <int>        zero samples for gradient" << std::endl;
  out << "  --oversample <float> oversample factor for zero sampling"
      << std::endl;
  out << "  --fnzw <float>     nonzero sample weight for f-est" << std::endl;
  out << "  --fzw <float>      zero sample weight for f-est" << std::endl;
  out << "  --gnzw <float>     nonzero sample weight for gradient" << std::endl;
  out << "  --gzw <float>      zero sample weight for gradient" << std::endl;
  out << "  --hash             compute hash map for zero sampling" << std::endl;
  out << "  --bulk-factor <int> factor for bulk zero sampling" << std::endl;
  out << "  --fuse             fuse gradient sampling and MTTKRP" << std::endl;
  out << "  --fuse-sa          fuse with sparse array gradient" << std::endl;
  out << "  --fit              compute fit metric" << std::endl;
  out << "  --step <type>      GCP-SGD optimization step type: ";
  for (unsigned i=0; i<Genten::GCP_Step::num_types; ++i) {
    out << Genten::GCP_Step::names[i];
    if (i != Genten::GCP_Step::num_types-1)
      out << ", ";
  }
  out << "  --adam-beta1       Decay rate for 1st moment avg." << std::endl;
  out << "  --adam-beta2       Decay rate for 2nd moment avg." << std::endl;
  out << "  --adam-eps         Shift in ADAM step." << std::endl;
  out << "  --async            Asynchronous SGD solver" << std::endl;
}

void Genten::AlgParams::print(std::ostream& out)
{
  out << "Generic options: " << std::endl;
  out << "  exec-space = " << Genten::Execution_Space::names[exec_space] << std::endl;
  out << "  method = " << Genten::Solver_Method::names[method] << std::endl;
  out << "  rank = " << rank << std::endl;
  out << "  seed = " << seed << std::endl;
  out << "  prng = " << (prng ? "true" : "false") << std::endl;
  out << "  maxiters = " << maxiters << std::endl;
  out << "  maxsecs = " << maxsecs << std::endl;
  out << "  tol = " << tol << std::endl;
  out << "  printitn = " << printitn << std::endl;
  out << "  debug = " << (debug ? "true" : "false") << std::endl;
  out << "  timings = " << (timings ? "true" : "false") << std::endl;
  out << "  full-gram = " << (full_gram ? "true" : "false") << std::endl;
  out << "  rank-def-solver = " << (rank_def_solver ? "true" : "false") << std::endl;
  out << "  rcond = " << rcond << std::endl;
  out << "  penalty = " << penalty << std::endl;
  out << "  dist-guess = " << dist_guess_method << std::endl;

  out << std::endl;
  out << "MTTKRP options:" << std::endl;
  out << "  mttkrp-method = " << Genten::MTTKRP_Method::names[mttkrp_method]
      << std::endl;
  out << "  mttkrp-all-method = " << Genten::MTTKRP_All_Method::names[mttkrp_all_method]
      << std::endl;
  out << "  mttkrp-nnz-tile-size = " << mttkrp_nnz_tile_size << std::endl;
  out << "  mttkrp-duplicated-tile-size = " << mttkrp_duplicated_factor_matrix_tile_size << std::endl;
  out << "  mttkrp-duplicated-threshold = " << mttkrp_duplicated_threshold << std::endl;
  out << "  warmup = " << (warmup ? "true" : "false") << std::endl;

  out << std::endl;
  out << "TTM options:" << std::endl;
  out << "  ttm-method = " << Genten::TTM_Method::names[ttm_method]
      << std::endl;

  out << std::endl;
  out << "CP-Opt options:" << std::endl;
  out << "  opt = " << Genten::Opt_Method::names[opt_method] << std::endl;
  out << "  lower = " << lower << std::endl;
  out << "  upper = " << upper << std::endl;
  out << "  rol = " << rolfilename << std::endl;
  out <<   "factr = " << factr << std::endl;
  out <<   "pgtol = " << pgtol << std::endl;
  out <<   "memory = " << memory << std::endl;
  out <<   "total-iters = " << max_total_iters << std::endl;
  out << "  hessian = " << Genten::Hess_Vec_Method::names[hess_vec_method] << std::endl;

  out << std::endl;
  out << "GCP options:" << std::endl;
  out << "  type = " << Genten::GCP_LossFunction::names[loss_function_type]
      << std::endl;
  out <<   "eps = " << loss_eps << std::endl;
  out <<   "gcp-tol = " << gcp_tol << std::endl;

  out << std::endl;
  out << "GCP-SGD options:" << std::endl;
  out << "  sampling = " << Genten::GCP_Sampling::names[sampling_type]
      << std::endl;
  out << "  rate = " << rate << std::endl;
  out << "  decay = " << decay << std::endl;
  out << "  fails = " << max_fails << std::endl;
  out << "  epochiters = " << epoch_iters << std::endl;
  out << "  frozeniters = " << frozen_iters << std::endl;
  out << "  rngiters = " << rng_iters << std::endl;
  out << "  fnzs = " << num_samples_nonzeros_value << std::endl;
  out << "  fzs = " << num_samples_zeros_value << std::endl;
  out << "  gnzs = " << num_samples_nonzeros_grad << std::endl;
  out << "  gzs = " << num_samples_zeros_grad << std::endl;
  out << "  oversample = " << oversample_factor << std::endl;
  out << "  fnzw = " << w_f_nz << std::endl;
  out << "  fzw = " << w_f_z << std::endl;
  out << "  gnzw = " << w_g_nz << std::endl;
  out << "  gzw = " << w_g_z << std::endl;
  out << "  bulk-factor = " << bulk_factor << std::endl;
  out << "  hash = " << (hash ? "true" : "false") << std::endl;
  out << "  fuse = " << (fuse ? "true" : "false") << std::endl;
  out << "  fuse-sa = " << (fuse_sa ? "true" : "false") << std::endl;
  out << "  fit = " << (compute_fit ? "true" : "false") << std::endl;
  out << "  step = " << Genten::GCP_Step::names[step_type] << std::endl;
  out << "  adam-beta1 = " << adam_beta1 << std::endl;
  out << "  adam-beta2 = " << adam_beta2 << std::endl;
  out << "  adam-eps = " << adam_eps << std::endl;
  out << "  async = " << (async ? "true" : "false") << std::endl;
  out << "  anneal = " << (anneal ? "true" : "false") << std::endl;
  if(anneal){
    out << "  anneal-min-lr = " << anneal_min_lr << std::endl;
    out << "  anneal-max-lr = " << anneal_max_lr << std::endl;
  }
}

ttb_real
Genten::parse_ttb_real(std::vector<std::string>& args,
                       const std::string& cl_arg,
                       ttb_real default_value, ttb_real min, ttb_real max)
{
  ttb_real tmp = default_value;
  auto it = std::find(args.begin(), args.end(), cl_arg);
  // If not found, try removing the '--'
  if ((it == args.end()) && (cl_arg.size() > 2) &&
      (cl_arg[0] == '-') && (cl_arg[1] == '-')) {
    it = std::find(args.begin(), args.end(), cl_arg.substr(2));
  }
  if (it != args.end()) {
    auto arg_it = it;
    // get next cl_arg
    ++it;
    if (it == args.end()) {
      args.erase(arg_it);
      return tmp;
    }
    // convert to ttb_real
    char *cend = 0;
    tmp = std::strtod(it->c_str(),&cend);
    // check if cl_arg is actually a ttb_real
    if (it->c_str() == cend) {
      std::ostringstream error_string;
      error_string << "Unparseable input: " << cl_arg << " " << *it
                   << ", must be a double" << std::endl;
      Genten::error(error_string.str());
      exit(1);
    }
    // Remove argument from list
    args.erase(arg_it, ++it);
  }
  // check if ttb_real is within bounds
  if (tmp < min || tmp > max) {
    std::ostringstream error_string;
    error_string << "Bad input: " << cl_arg << " " << tmp
                 << ",  must be in the range (" << min << ", " << max
                 << ")" << std::endl;
    Genten::error(error_string.str());
    exit(1);
  }
  return tmp;
}

ttb_indx
Genten::parse_ttb_indx(std::vector<std::string>& args,
                       const std::string& cl_arg,
                       ttb_indx default_value, ttb_indx min, ttb_indx max)
{
  ttb_indx tmp = default_value;
  auto it = std::find(args.begin(), args.end(), cl_arg);
  // If not found, try removing the '--'
  if ((it == args.end()) && (cl_arg.size() > 2) &&
      (cl_arg[0] == '-') && (cl_arg[1] == '-')) {
    it = std::find(args.begin(), args.end(), cl_arg.substr(2));
  }
  if (it != args.end()) {
    auto arg_it = it;
    // get next cl_arg
    ++it;
    if (it == args.end()) {
      args.erase(arg_it);
      return tmp;
    }
    // convert to ttb_indx
    char *cend = 0;
    tmp = std::strtol(it->c_str(),&cend,10);
    // check if cl_arg is actually a ttb_indx
    if (it->c_str() == cend) {
      std::ostringstream error_string;
      error_string << "Unparseable input: " << cl_arg << " " << *it
                   << ", must be an integer" << std::endl;
      Genten::error(error_string.str());
      exit(1);
    }
    // Remove argument from list
    args.erase(arg_it, ++it);
  }
  // check if ttb_real is within bounds
  if (tmp < min || tmp > max) {
    std::ostringstream error_string;
    error_string << "Bad input: " << cl_arg << " " << tmp
                 << ",  must be in the range (" << min << ", " << max
                 << ")" << std::endl;
    Genten::error(error_string.str());
    exit(1);
  }
  return tmp;
}

ttb_bool
Genten::parse_ttb_bool(std::vector<std::string>& args,
                       const std::string& cl_arg_on,
                       const std::string& cl_arg_off,
                       ttb_bool default_value)
{
  // return true if arg_on is found
  auto it = std::find(args.begin(), args.end(), cl_arg_on);
  // If not found, try removing the '--'
  if ((it == args.end()) && (cl_arg_on.size() > 2) &&
      (cl_arg_on[0] == '-') && (cl_arg_on[1] == '-')) {
    it = std::find(args.begin(), args.end(), cl_arg_on.substr(2));
  }
  if (it != args.end()) {
    args.erase(it);
    return true;
  }

  // return false if arg_off is found
  it = std::find(args.begin(), args.end(), cl_arg_off);
  // If not found, try removing the '--'
  if ((it == args.end()) && (cl_arg_off.size() > 2) &&
      (cl_arg_off[0] == '-') && (cl_arg_off[1] == '-')) {
    it = std::find(args.begin(), args.end(), cl_arg_off.substr(2));
  }
  if (it != args.end()) {
    args.erase(it);
    return false;
  }

  // return default value if not specified on command line
  return default_value;
}

std::string
Genten::parse_string(std::vector<std::string>& args, const std::string& cl_arg,
                     const std::string& default_value)
{
  std::string tmp = default_value;
  auto it = std::find(args.begin(), args.end(), cl_arg);
  // If not found, try removing the '--'
  if ((it == args.end()) && (cl_arg.size() > 2) &&
      (cl_arg[0] == '-') && (cl_arg[1] == '-')) {
    it = std::find(args.begin(), args.end(), cl_arg.substr(2));
  }
  if (it != args.end()) {
    auto arg_it = it;
    // get next cl_arg
    ++it;
    if (it == args.end()) {
      args.erase(arg_it);
      return tmp;
    }
    // get argument
    tmp = *it;
    // Remove argument from list
    args.erase(arg_it, ++it);
  }
  return tmp;
}

Genten::IndxArray
Genten::parse_ttb_indx_array(std::vector<std::string>& args,
                             const std::string& cl_arg,
                             const Genten::IndxArray& default_value,
                             ttb_indx min, ttb_indx max)
{
  char *cend = 0;
  ttb_indx tmp;
  std::vector<ttb_indx> vals;
  auto it = std::find(args.begin(), args.end(), cl_arg);
  // If not found, try removing the '--'
  if ((it == args.end()) && (cl_arg.size() > 2) &&
      (cl_arg[0] == '-') && (cl_arg[1] == '-')) {
    it = std::find(args.begin(), args.end(), cl_arg.substr(2));
  }
  if (it != args.end()) {
    auto arg_it = it;
    // get next cl_arg
    ++it;
    if (it == args.end()) {
      args.erase(arg_it);
      return default_value;
    }
    const char *arg_val = it->c_str();
    if (arg_val[0] != '[') {
      std::ostringstream error_string;
      error_string << "Unparseable input: " << cl_arg << " " << arg_val
                   << ", must be of the form [int,...,int] with no spaces"
                   << std::endl;
      Genten::error(error_string.str());
      exit(1);
    }
    while (strlen(arg_val) > 0 && arg_val[0] != ']') {
      ++arg_val; // Move past ,
      // convert to ttb_indx
      tmp = std::strtol(arg_val,&cend,10);
      // check if cl_arg is actually a ttb_indx
      if (arg_val == cend) {
        std::ostringstream error_string;
        error_string << "Unparseable input: " << cl_arg << " " << arg_val
                     << ", must be of the form [int,...,int] with no spaces"
                     << std::endl;
        Genten::error(error_string.str());
        exit(1);
      }
      // check if ttb_indx is within bounds
      if (tmp < min || tmp > max) {
        std::ostringstream error_string;
        error_string << "Bad input: " << cl_arg << " " << arg_val
                     << ",  must be in the range (" << min << ", " << max
                     << ")" << std::endl;
        Genten::error(error_string.str());
        exit(1);
      }
      vals.push_back(tmp);
      arg_val = cend;
    }
    // Remove argument from list
    args.erase(arg_it, ++it);
    // return index array if everything is OK
    return Genten::IndxArray(vals.size(), vals.data());
  }
  // return default value if not specified on command line
  return default_value;
}

void
Genten::parse_ptree_value(const Genten::ptree& input, const std::string& name,
                          bool& val)
{
  val = input.get<bool>(name, val);
}

void
Genten::parse_ptree_value(const Genten::ptree& input, const std::string& name,
                          std::string& val)
{
  val = input.get<std::string>(name, val);
}

std::vector<std::string>
Genten::build_arg_list(int argc, char** argv)
{
  std::vector<std::string> arg_list(argc-1);
  for (int i=1; i<argc; ++i)
    arg_list[i-1] = argv[i];
  return arg_list;
}

bool
Genten::check_and_print_unused_args(const std::vector<std::string>& args,
                                    std::ostream& out)
{
  if (args.size() == 0)
    return false;

  out << std::endl << "Error!  Unknown command line arguments: ";
  for (auto arg : args)
    out << arg << " ";
  out << std::endl << std::endl;
  return true;
}
