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

Genten::AlgParams::AlgParams() :
  method("cp-als"),
  rank(16),
  seed(12345),
  prng(false),
  maxiters(100),
  maxsecs(-1.0),
  tol(0.0004),
  printitn(1),
  debug(false),
  mttkrp_method(MTTKRP_Method::default_type),
  mttkrp_duplicated_factor_matrix_tile_size(0),
  loss_function_type(Genten::GCP_LossFunction::default_type),
  loss_eps(1.0e-10),
  rolfilename(""),
  rate(1.0e-3),
  decay(0.1),
  max_fails(1),
  epoch_iters(1000),
  num_samples_nonzeros_value(0),
  num_samples_zeros_value(0),
  num_samples_nonzeros_grad(0),
  num_samples_zeros_grad(0)
{}

void Genten::AlgParams::parse(int argc, char* argv[])
{
  // Parse options from command-line, using default values set above as defaults

  // Generic options
  method = parse_string(argc, argv, "--method", method.c_str());
  rank = parse_ttb_indx(argc, argv, "--rank", rank, 1, INT_MAX);
  seed = parse_ttb_indx(argc, argv, "--seed", seed, 0, INT_MAX);
  prng = parse_ttb_bool(argc, argv, "--prng", prng);
  maxiters = parse_ttb_indx(argc, argv, "--maxiters", maxiters, 1, INT_MAX);
  maxsecs = parse_ttb_real(argc, argv, "--maxsecs", maxsecs, -1.0, DOUBLE_MAX);
  tol = parse_ttb_real(argc, argv, "--tol", tol, 0.0, 1.0);
  printitn = parse_ttb_indx(argc, argv, "--printitn", printitn, 0, INT_MAX);
  debug = parse_ttb_bool(argc, argv, "--debug", debug);

  // MTTKRP options
  mttkrp_method = parse_ttb_enum(argc, argv, "--mttkrp_method", mttkrp_method,
                                 Genten::MTTKRP_Method::num_types,
                                 Genten::MTTKRP_Method::types,
                                 Genten::MTTKRP_Method::names);
  mttkrp_duplicated_factor_matrix_tile_size =
    parse_ttb_indx(argc, argv, "--mttkrp_tile_size",
                   mttkrp_duplicated_factor_matrix_tile_size, 0, INT_MAX);
  warmup = parse_ttb_bool(argc, argv, "--warmup", warmup);

  // GCP options
  loss_function_type = parse_ttb_enum(argc, argv, "--type", loss_function_type,
                                      Genten::GCP_LossFunction::num_types,
                                      Genten::GCP_LossFunction::types,
                                      Genten::GCP_LossFunction::names);
  loss_eps = parse_ttb_real(argc, argv, "--eps", loss_eps, 0.0, 1.0);

  // GCP-Opt options
  rolfilename = parse_string(argc, argv, "--rol", rolfilename.c_str());

  // GCP-SGD options
  rate = parse_ttb_real(argc, argv, "--rate", rate, 0.0, DOUBLE_MAX);
  decay = parse_ttb_real(argc, argv, "--decay", decay, 0.0, 1.0);
  max_fails = parse_ttb_indx(argc, argv, "--fails", max_fails, 0, INT_MAX);
  epoch_iters =
    parse_ttb_indx(argc, argv, "--epochiters", epoch_iters, 0, INT_MAX);
  num_samples_nonzeros_value =
    parse_ttb_indx(argc, argv, "--fnzs", num_samples_nonzeros_value, 0, INT_MAX);
  num_samples_zeros_value =
    parse_ttb_indx(argc, argv, "--fzs", num_samples_zeros_value, 0, INT_MAX);
  num_samples_nonzeros_grad =
    parse_ttb_indx(argc, argv, "--gnzs", num_samples_nonzeros_grad, 0, INT_MAX);
  num_samples_zeros_grad =
    parse_ttb_indx(argc, argv, "--gzs", num_samples_zeros_grad, 0, INT_MAX);
}

void Genten::AlgParams::print_help(std::ostream& out)
{
  out << "Generic options: " << std::endl;
  out << "  --method <string>  decomposition method" << std::endl;
  out << "  --rank <int>       rank of factorization to compute" << std::endl;
  out << "  --seed <int>       seed for random number generator used in initial guess" << std::endl;
  out << "  --prng             use parallel random number generator (not consistent with Matlab)" << std::endl;
  out << "  --maxiters <int>   maximum iterations to perform" << std::endl;
  out << "  --maxsecs <float>  maximum running time" << std::endl;
  out << "  --tol <float>      stopping tolerance" << std::endl;
  out << "  --printitn <int>   print every <int>th iteration; 0 for no printing" << std::endl;
  out << "  --debug            turn on debugging output" << std::endl;

  out << std::endl;
  out << "MTTKRP options:" << std::endl;
  out << "  --mttkrp_method <method> MTTKRP algorithm: ";
  for (unsigned i=0; i<Genten::MTTKRP_Method::num_types; ++i) {
    out << Genten::MTTKRP_Method::names[i];
    if (i != Genten::MTTKRP_Method::num_types-1)
      out << ", ";
  }
  out << std::endl;
  out << "  --mttkrp_tile_size <int> tile size for mttkrp algorithm"
      << std::endl;
  out << "  --warmup           do an iteration of mttkrp to warmup (useful for generating accurate timing information)" << std::endl;

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

  out << std::endl;
  out << "GCP-Opt options:" << std::endl;
  out << "  --rol <string>     path to ROL optimization settings file for GCP method" << std::endl;

  out << std::endl;
  out << "GCP-SGD options:" << std::endl;
  out << "  --rate <float>     initial step size" << std::endl;
  out << "  --decay <float>    rate step size decreases on fails" << std::endl;
  out << "  --fails <int>      maximum number of fails" << std::endl;
  out << "  --epochiters <int> iterations per epoch" << std::endl;
  out << "  --fnzs <int>       nonzero samples for f-est" << std::endl;
  out << "  --fzs <int>        zero samples for f-est" << std::endl;
  out << "  --gnzs <int>       nonzero samples for gradient" << std::endl;
  out << "  --gzs <int>        zero samples for gradient" << std::endl;
}

void Genten::AlgParams::print(std::ostream& out)
{
  out << "Generic options: " << std::endl;
  out << "  method = " << method << std::endl;
  out << "  rank = " << rank << std::endl;
  out << "  seed = " << seed << std::endl;
  out << "  prng = " << (prng ? "true" : "false") << std::endl;
  out << "  maxiters = " << maxiters << std::endl;
  out << "  maxsecs = " << maxsecs << std::endl;
  out << "  tol = " << tol << std::endl;
  out << "  printitn = " << printitn << std::endl;
  out << "  debug = " << (debug ? "true" : "false") << std::endl;

  out << std::endl;
  out << "MTTKRP options:" << std::endl;
  out << "  mttkrp_method = " << Genten::MTTKRP_Method::types[mttkrp_method]
      << std::endl;
  out << "  mttkrp_tile_size = " << mttkrp_duplicated_factor_matrix_tile_size
      << std::endl;
  out << "  warmup = " << (warmup ? "true" : "false") << std::endl;

  out << std::endl;
  out << "GCP options:" << std::endl;
  out << "  type = " << Genten::GCP_LossFunction::names[loss_function_type]
      << std::endl;
  out <<   "eps = " << loss_eps << std::endl;

  out << std::endl;
  out << "GCP-Opt options:" << std::endl;
  out << "  rol = " << rolfilename << std::endl;

   out << std::endl;
  out << "GCP-SGD options:" << std::endl;
  out << "  rate = " << rate << std::endl;
  out << "  decay = " << decay << std::endl;
  out << "  fails = " << max_fails << std::endl;
  out << "  epochiters = " << epoch_iters << std::endl;
  out << "  fnzs = " << num_samples_nonzeros_value << std::endl;
  out << "  fzs = " << num_samples_zeros_value << std::endl;
  out << "  gnzs = " << num_samples_nonzeros_grad << std::endl;
  out << "  gzs = " << num_samples_zeros_grad << std::endl;
}

ttb_real
Genten::parse_ttb_real(int argc, char** argv, std::string cl_arg,
                       ttb_real default_value, ttb_real min, ttb_real max)
{
  int arg=1;
  char *cend = 0;
  ttb_real tmp;
  while (arg < argc) {
    if (cl_arg == std::string(argv[arg])) {
      // get next cl_arg
      arg++;
      if (arg >= argc)
        return default_value;
      // convert to ttb_real
      tmp = std::strtod(argv[arg],&cend);
      // check if cl_arg is actuall a ttb_real
      if (argv[arg] == cend) {
        std::ostringstream error_string;
        error_string << "Unparseable input: " << cl_arg << " " << argv[arg]
                     << ", must be a double" << std::endl;
        Genten::error(error_string.str());
        exit(1);
        // check if ttb_real is within bounds
      }
      if (tmp < min || tmp > max) {
        std::ostringstream error_string;
        error_string << "Bad input: " << cl_arg << " " << argv[arg]
                     << ",  must be in the range (" << min << ", " << max
                     << ")" << std::endl;
        Genten::error(error_string.str());
        exit(1);
      }
      // return ttb_real if everything is OK
      return tmp;
    }
    arg++;
  }
  // return default value if not specified on command line
  return default_value;
}

ttb_indx
Genten::parse_ttb_indx(int argc, char** argv, std::string cl_arg,
                       ttb_indx default_value, ttb_indx min, ttb_indx max)
{
  int arg=1;
  char *cend = 0;
  ttb_indx tmp;
  while (arg < argc) {
    if (cl_arg == std::string(argv[arg])) {
      // get next cl_arg
      arg++;
      if (arg >= argc)
        return default_value;
      // convert to ttb_real
      tmp = std::strtol(argv[arg],&cend,10);
      // check if cl_arg is actuall a ttb_real
      if (argv[arg] == cend) {
        std::ostringstream error_string;
        error_string << "Unparseable input: " << cl_arg << " " << argv[arg]
                     << ", must be an unsigned integer" << std::endl;
        Genten::error(error_string.str());
        exit(1);
        // check if ttb_real is within bounds
      }
      if (tmp < min || tmp > max) {
        std::ostringstream error_string;
        error_string << "Bad input: " << cl_arg << " " << argv[arg]
                     << ",  must be in the range (" << min << ", " << max
                     << ")" << std::endl;
        Genten::error(error_string.str());
        exit(1);
      }
      // return ttb_real if everything is OK
      return tmp;
    }
    arg++;
  }
  // return default value if not specified on command line
  return default_value;
}

ttb_bool
Genten::parse_ttb_bool(int argc, char** argv, std::string cl_arg,
                       ttb_bool default_value)
{
  int arg=1;
  while (arg < argc) {
    if (cl_arg == std::string(argv[arg])) {
      // return true if arg is found
      return true;
    }
    arg++;
  }
  // return default value if not specified on command line
  return default_value;
}

std::string
Genten::parse_string(int argc, char** argv, std::string cl_arg,
                     std::string default_value)
{
  int arg=1;
  std::string tmp;
  while (arg < argc) {
    if (cl_arg == std::string(argv[arg])) {
      // get next cl_arg
      arg++;
      if (arg >= argc)
        return "";
      // convert to string
      tmp = std::string(argv[arg]);
      // return ttb_real if everything is OK
      return tmp;
    }
    arg++;
  }
  // return default value if not specified on command line
  return default_value;

}

Genten::IndxArray
Genten::parse_ttb_indx_array(int argc, char** argv, std::string cl_arg,
                             const Genten::IndxArray& default_value,
                             ttb_indx min, ttb_indx max)
{
  int arg=1;
  char *cend = 0;
  ttb_indx tmp;
  std::vector<ttb_indx> vals;
  while (arg < argc) {
    if (cl_arg == std::string(argv[arg])) {
      // get next cl_arg
      arg++;
      if (arg >= argc)
        return default_value;
      char *arg_val = argv[arg];
      if (arg_val[0] != '[') {
        std::ostringstream error_string;
        error_string << "Unparseable input: " << cl_arg << " " << arg_val
                     << ", must be of the form { int, ... }" << std::endl;
        Genten::error(error_string.str());
        exit(1);
      }
      while (strlen(arg_val) > 0 && arg_val[0] != ']') {
        ++arg_val; // Move past ,
        // convert to ttb_real
        tmp = std::strtol(arg_val,&cend,10);
        // check if cl_arg is actuall a ttb_real
        if (arg_val == cend) {
          std::ostringstream error_string;
          error_string << "Unparseable input: " << cl_arg << " " << arg_val
                       << ", must be of the form { int, ... }" << std::endl;
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
      // return index array if everything is OK
      return Genten::IndxArray(vals.size(), vals.data());
    }
    arg++;
  }
  // return default value if not specified on command line
  return default_value;
}
