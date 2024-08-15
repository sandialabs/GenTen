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

#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "Genten_Util.hpp"
#include "Genten_IndxArray.hpp"
#include "Genten_Ptree.hpp"

#ifdef HAVE_PYTHON_EMBED
#include "pybind11/pytypes.h"
#endif

namespace Genten {

  // Struct for passing various algorithmic parameters
  struct __attribute__((visibility("default"))) AlgParams {

    // Generic options
    Execution_Space::type exec_space; // Chosen execution space
    IndxArray proc_grid; // User-defined processor grid
    bool sparse;         // Sparse (or dense) tensor
    Solver_Method::type method; // Solver method ("cp-als", "gcp-sgd", ...)
    ttb_indx rank;       // Rank of decomposition
    unsigned long seed;  // Random number seed for initial guess
    bool prng;           // Use parallel random number generator
    ttb_indx maxiters;   // Maximum number of iterations
    ttb_real maxsecs;    // Maximum amount of time
    ttb_real tol;        // Decomposition tolerance
    ttb_indx printitn;   // Print iterations
    bool debug;          // Print debugging info
    bool timings;        // Print accurate kernel timing info (requires fences)
    std::string timings_xml; // XML file to save timing info
    bool full_gram;      // Use full Gram matrix formulation
    bool rank_def_solver; // Use rank-deficient least-squares solver
    ttb_real rcond;      // Truncation threshold in rank-deficient solver
    ttb_real penalty;    // Regularization penalty
    std::string dist_guess_method; // Method for distributed initial guess
    bool scale_guess_by_norm_x; // Scale initial guess by norm of the tensor

    // MTTKRP options
    MTTKRP_Method::type mttkrp_method; // MTTKRP algorithm
    MTTKRP_All_Method::type mttkrp_all_method; // MTTKRP algorithm for all dims
    unsigned mttkrp_nnz_tile_size; // Nonzero tile size (i.e., RowBlockSize)
    unsigned mttkrp_duplicated_factor_matrix_tile_size; // Tile size for MTTKRP
    ttb_real mttkrp_duplicated_threshold;  // Theshold for when dup is used
    Dist_Update_Method::type dist_update_method;
    bool optimize_maps;  // Optimize Tpetra maps to reduce communication'
    bool build_maps_on_device;  //Build Tpetra maps on the device
    bool warmup; // Warmup by calling MTTKRP before decomposition

    // TTM options
    TTM_Method::type ttm_method; // TTM algorithm

    // CP-Opt options
    Opt_Method::type opt_method; // Optimization method
    ttb_real lower;              // Lower bound of factorization
    ttb_real upper;              // Upper bound of factorization
    std::string rolfilename;     // Filename for ROL solver options
    ttb_real ftol;               // residual tolerance for L-BFGS-B
    ttb_real gtol;               // gradient tolerance for L-BFGS-B
    ttb_indx memory;             // memory parameter for L-BFGS-B
    ttb_indx sub_iters;          // inner iterations for L-BFGS-B
    Hess_Vec_Method::type hess_vec_method; // Hessian-vector product method
    Hess_Vec_Tensor_Method::type hess_vec_tensor_method; // Hessian-vector product method for tensor-only term
    Hess_Vec_Prec_Method::type hess_vec_prec_method; // Preconditoning method for hessian-vector product

    // GCP options
    std::string loss_function_type;      // Loss function for GCP
    ttb_real loss_eps;                   // Perturbation for GCP
    ttb_real loss_param;                 // Generic parameter for loss functions
    ttb_real gcp_tol;                    // Tolerance for GCP algorithm
    GCP_Goal_Method::type goal_method;   // Way of supplying goal function
    std::string python_module_name;      // Module name for python goal
    std::string python_object_name;      // Object name for python module goal
#ifdef HAVE_PYTHON_EMBED
    pybind11::object python_object;      // Python object for python goal
#endif

    // GCP-SGD options
    GCP_Sampling::type sampling_type;    // Sampling type
    ttb_real rate;                       // Initial step size
    ttb_real decay;                      // Rate step size decreases on fails
    ttb_indx max_fails;                  // Maximum number of fails
    ttb_indx epoch_iters;                // Number of iterations per epoch
    ttb_indx frozen_iters;               // Number of iterations w/frozen grad
    ttb_indx rng_iters;                  // Number of loops in RNG
    unsigned long gcp_seed;              // Random number seed for GCP
    ttb_indx num_samples_nonzeros_value; // Nonzero samples for f-est
    ttb_indx num_samples_zeros_value;    // Zero sampels for f-est
    ttb_indx num_samples_nonzeros_grad;  // Nonzero samples for gradient
    ttb_indx num_samples_zeros_grad;     // Zero samples for gradient
    ttb_real oversample_factor;          // Factor for oversampling of zeros
    ttb_indx bulk_factor;                // Factor for bulk sampling
    ttb_real w_f_nz;                     // Nonzero sample weight for f
    ttb_real w_f_z;                      // Zero sample weight for f
    ttb_real w_g_nz;                     // Nonzero sample weight for grad
    ttb_real w_g_z;                      // Zero sample weight for grad
    bool normalize;                      // Normalize initial Ktensor
    bool hash;                           // Hash tensor instead of sorting
    bool fuse;                           // Fuse sampling and gradient kernels
    bool fuse_sa;                        // Fused with sparse array gradient
    bool compute_fit;                    // Compute fit metric
    GCP_Step::type step_type;            // GCP-SGD step type
    ttb_real adam_beta1;                 // Decay rate of first moment avg.
    ttb_real adam_beta2;                 // Decay rate of second moment avg.
    ttb_real adam_eps;                   // Shift in ADAM/AdaGrad step
    bool async;                          // Asynchronous SGD solver
    GCP_AnnealerMethod::type annealer;   // Annealer method
    ttb_real anneal_min_lr;              // Cosine annealer min learning rate
    ttb_real anneal_max_lr;              // Cosine annealer max learning rate
    ttb_real anneal_Ti;                  // Cosine annealer initial temperature

    // GCP Federated learning options
    GCP_FedMethod::type fed_method;      // Learning method (FedOpt or FedAvg)
    GCP_Step::type meta_step_type;       // Meta step type
    ttb_real meta_rate;                  // Initial meta step size
    ttb_indx downpour_iters;             // Downpour iterations

    // Streaming GCP options
    GCP_Streaming_Solver::type streaming_solver;  // Streaming solver
    GCP_Streaming_History_Method::type history_method; // History method
    GCP_Streaming_Window_Method::type window_method; // Windowing method
    ttb_indx window_size;                // Number of terms in window
    ttb_real window_weight;              // Multiplier for each window term
    ttb_real window_penalty;             // Multiplier for entire window
    ttb_real factor_penalty;             // Penalty term on factor matrices

    // Constructor initializing values to defaults
    AlgParams();

    // Parse options
    void parse(std::vector<std::string>& args);
    void parse(const ptree& tree);

    // Print help string
    static void print_help(std::ostream& out);

    // Print parmeters
    void print(std::ostream& out) const;

    // Fixup alg params to correct values
    template <typename ExecSpace>
    void fixup(std::ostream& out);
  };

  ttb_real parse_ttb_real(std::vector<std::string>& args, const std::string& cl_arg, ttb_real default_value, ttb_real min=0.0, ttb_real max=1.0);
  ttb_indx parse_ttb_indx(std::vector<std::string>& args, const std::string& cl_arg, ttb_indx default_value, ttb_indx min=0, ttb_indx max=100);
  ttb_bool parse_ttb_bool(std::vector<std::string>& args, const std::string& cl_arg_on, const std::string& cl_off_off, ttb_bool default_value);
  std::string parse_string(std::vector<std::string>& args, const std::string& cl_arg, const std::string& default_value);
  IndxArray parse_ttb_indx_array(std::vector<std::string>& args, const std::string& cl_arg, const IndxArray& default_value, ttb_indx min=1, ttb_indx max=INT_MAX);
  template <typename T>
  T parse_ttb_enum(std::vector<std::string>& args, const std::string& cl_arg,
                   T default_value, unsigned num_values, const T* values,
                   const char*const* names)
  {
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
      // convert to string
      std::string arg_val = *it;
      // Remove argument from list
      args.erase(arg_it, ++it);
      // find name in list of names
      for (unsigned i=0; i<num_values; ++i) {
        if (arg_val == names[i])
          return values[i];
      }
      // if we got here, name wasn't found
      std::ostringstream error_string;
      error_string << "Bad input: " << cl_arg << " " << arg_val << ",  must be one of the values: ";
      for (unsigned i=0; i<num_values; ++i) {
        error_string << names[i];
        if (i != num_values-1)
          error_string << ", ";
      }
      error_string << "." << std::endl;
      Genten::error(error_string.str());
      exit(1);
    }
    // return default value if not specified on command line
    return default_value;
  }

  template <typename T>
  typename T::type parse_enum(const std::string& name) {
    for (unsigned i=0; i<T::num_types; ++i) {
      if (name == T::names[i])
        return T::types[i];
    }

    // if we got here, name wasn't found
    std::ostringstream error_string;
    error_string << "Invalid enum choice " << name
                 << ",  must be one of the values: ";
    for (unsigned i=0; i<T::num_types; ++i) {
      error_string << T::names[i];
      if (i != T::num_types-1)
        error_string << ", ";
    }
    error_string << "." << std::endl;
    Genten::error(error_string.str());
    return T::default_type;
  }

  // A helper function for parsing standard values out of a ptree
  // where the default value is the current value in val
  template <typename T, typename U, typename V>
  void parse_ptree_value(const Genten::ptree& input, const std::string& name,
                         T& val, const U& lower, const V& upper)
  {
    val = input.get<T>(name, val);
    if (val < T(lower) || val > T(upper)) {
      std::ostringstream error_string;
      error_string << "Bad input: " << name << " " << val
                   << ",  must be in the range (" << lower << ", " << upper
                   << ")" << std::endl;
      Genten::error(error_string.str());
    }
  }
  void parse_ptree_value(const Genten::ptree& input, const std::string& name,
                         bool& val);
  void parse_ptree_value(const Genten::ptree& input, const std::string& name,
                         std::string& val);

  template <typename T, typename U, typename V>
  void parse_ptree_value(const Genten::ptree& input, const std::string& name,
                         std::vector<T>& val,
                         const U& lower, const V& upper)
  {
    val = input.get<std::vector<T> >(name);
    for (const auto& v : val) {
      if (v < T(lower) || v > T(upper)) {
        std::ostringstream error_string;
        error_string << "Bad input: " << name << " " << v
                     << ",  must be in the range (" << lower << ", " << upper
                     << ")" << std::endl;
        Genten::error(error_string.str());
      }
    }
  }

  // A helper function for parsing an enum out of a ptree where T is the
  // struct using the pattern in Genten_Util.hpp
  template <typename T>
  void parse_ptree_enum(const Genten::ptree& input, const std::string& name,
                        typename T::type& val)
  {
    val = Genten::parse_enum<T>(input.get<std::string>(name, T::names[val]));
  }
  template <typename T>
  typename T::type parse_ptree_enum(const Genten::ptree& input, const std::string& name)
  {
    return Genten::parse_enum<T>(input.get<std::string>(name, T::names[T::default_type]));
  }

  // Convert (argc,argv) to list of strings
  std::vector<std::string> build_arg_list(int argc, char** argv);

  // Print out unrecognized command line arguments.  Returns true if there
  // are any, false otherwise
  bool check_and_print_unused_args(const std::vector<std::string>& args,
                                   std::ostream& out);

  template <typename ExecSpace>
  void AlgParams::fixup(std::ostream& out) {
    typedef SpaceProperties<ExecSpace> space_prop;

    // Even for dense tensors we use sparse mttkrp for GCP-SGD/GCP-FED
    const bool use_sparse = sparse || method == Solver_Method::GCP_SGD ||
      method == Solver_Method::GCP_FED;

    // Compute default MTTKRP method
    if (mttkrp_method == MTTKRP_Method::Default && use_sparse) {

      // Always use Single if there is only a single thread
      if (space_prop::concurrency() == 1)
        mttkrp_method = MTTKRP_Method::Single;

      // Use Atomic on Cuda if it supports fast atomics for ttb_real.
      // This is true with float on all arch's or float/double on Pascal (6.0)
      // or later
      else if (space_prop::is_cuda && (space_prop::cuda_arch() >= 600 ||
                                       sizeof(ttb_real) == 4))
        mttkrp_method = MTTKRP_Method::Atomic;

      else if (space_prop::is_cuda)
        mttkrp_method = MTTKRP_Method::Perm;

      else if (space_prop::is_hip)
        mttkrp_method = MTTKRP_Method::Atomic;

      else if (space_prop::is_sycl)
        mttkrp_method = MTTKRP_Method::Atomic;

      // Otherwise use Perm or Duplicated on CPU depending on the method
      else {
        if (method == Solver_Method::GCP_SGD)
          mttkrp_method = MTTKRP_Method::Duplicated;
        else
          mttkrp_method = MTTKRP_Method::Perm;
      }
    }
    else if (mttkrp_method == MTTKRP_Method::Default && !use_sparse) {
      mttkrp_method = MTTKRP_Method::RowBased;
    }

    // Compute default MTTKRP-All method
    if (mttkrp_all_method == MTTKRP_All_Method::Default && use_sparse) {

      // Always use Single if there is only a single thread
      if (space_prop::concurrency() == 1)
        mttkrp_all_method = MTTKRP_All_Method::Single;

      // Always use atomic on Cuda if fused
      else if (space_prop::is_cuda &&
               method == Solver_Method::GCP_SGD &&
               sampling_type == GCP_Sampling::SemiStratified && fuse)
        mttkrp_all_method = MTTKRP_All_Method::Atomic;

      // Use Atomic on Cuda if it supports fast atomics for ttb_real.
      // This is true with float on all arch's or float/double on Pascal (6.0)
      // or later
      else if (space_prop::is_cuda && (space_prop::cuda_arch() >= 600 ||
                                       sizeof(ttb_real) == 4))
        mttkrp_all_method = MTTKRP_All_Method::Atomic;

      else if (space_prop::is_cuda)
        mttkrp_all_method = MTTKRP_All_Method::Iterated;

      else if (space_prop::is_hip)
        mttkrp_all_method = MTTKRP_All_Method::Atomic;

      else if (space_prop::is_sycl)
        mttkrp_all_method = MTTKRP_All_Method::Atomic;

      // Otherwise use Iterated or Duplicated depending on the method
      else {
        if (method == Solver_Method::GCP_SGD)
          mttkrp_all_method = MTTKRP_All_Method::Duplicated;
        else
          mttkrp_all_method = MTTKRP_All_Method::Iterated;
      }
    }
    else if (mttkrp_all_method == MTTKRP_All_Method::Default && !use_sparse) {
      mttkrp_all_method = MTTKRP_All_Method::Iterated;
    }

    // Compute default hess-vec-tensor method
    if (hess_vec_tensor_method == Hess_Vec_Tensor_Method::Default) {

      // Always use Single if there is only a single thread
      if (space_prop::concurrency() == 1)
        hess_vec_tensor_method = Hess_Vec_Tensor_Method::Single;

      // Use Atomic on Cuda if it supports fast atomics for ttb_real.
      // This is true with float on all arch's or float/double on Pascal (6.0)
      // or later
      else if (space_prop::is_cuda && (space_prop::cuda_arch() >= 600 ||
                                  sizeof(ttb_real) == 4))
        hess_vec_tensor_method = Hess_Vec_Tensor_Method::Atomic;

      else if (space_prop::is_sycl)
        hess_vec_tensor_method = Hess_Vec_Tensor_Method::Atomic;

      else
        hess_vec_tensor_method = Hess_Vec_Tensor_Method::Perm;
    }

    // Fix invalid choices from the user:
    //   * Single and Duplicated are not valid on Cuda
    //   * Atomic is required for fused GCP-SGD with Semi-Stratified sampling on
    //     Cuda
    if (space_prop::is_cuda) {
      if (mttkrp_method == MTTKRP_Method::Single ||
          mttkrp_method == MTTKRP_Method::Duplicated) {
        out << "MTTKRP method " << MTTKRP_Method::names[mttkrp_method]
            << " is invalid for Cuda, changing to ";
        if (space_prop::cuda_arch() >= 600 || sizeof(ttb_real) == 4)
          mttkrp_method = MTTKRP_Method::Atomic;
        else
          mttkrp_method = MTTKRP_Method::Perm;
        out << MTTKRP_Method::names[mttkrp_method] << "." << std::endl;
      }
      if (mttkrp_all_method == MTTKRP_All_Method::Single ||
          mttkrp_all_method == MTTKRP_All_Method::Duplicated) {
        out << "MTTKRP-All method "
            << MTTKRP_All_Method::names[mttkrp_all_method]
            << " is invalid for Cuda, changing to ";
        if (space_prop::cuda_arch() >= 600 || sizeof(ttb_real) == 4 ||
            (method == Solver_Method::GCP_SGD &&
             sampling_type == GCP_Sampling::SemiStratified && fuse))
          mttkrp_all_method = MTTKRP_All_Method::Atomic;
        else
          mttkrp_all_method = MTTKRP_All_Method::Iterated;
        out << MTTKRP_All_Method::names[mttkrp_all_method] << "." << std::endl;
      }
      if (method == Solver_Method::GCP_SGD &&
          sampling_type == GCP_Sampling::SemiStratified &&
          fuse && mttkrp_all_method != MTTKRP_All_Method::Atomic) {
        mttkrp_all_method = MTTKRP_All_Method::Atomic;
        out << "Fused semi-stratified sampling/MTTKRP method requires atomic"
            << " on Cuda.  Changing MTTKRP-All method to atomic." << std::endl;
      }
      if (method == Solver_Method::CP_OPT &&
          hess_vec_method == Hess_Vec_Method::Full &&
          (hess_vec_tensor_method == Hess_Vec_Tensor_Method::Single ||
           hess_vec_tensor_method == Hess_Vec_Tensor_Method::Duplicated)) {
        out << "hess-vec tensor method " << Hess_Vec_Tensor_Method::names[hess_vec_tensor_method]
            << " is invalid on Cuda.  Changing method to atomic." << std::endl;
        hess_vec_tensor_method = Hess_Vec_Tensor_Method::Atomic;
      }
    } else if (space_prop::is_hip) {
      if (mttkrp_method == MTTKRP_Method::Single ||
          mttkrp_method == MTTKRP_Method::Duplicated) {
        out << "MTTKRP method " << MTTKRP_Method::names[mttkrp_method]
            << " is invalid for HIP, changing to ";
        mttkrp_method = MTTKRP_Method::Atomic;
        out << MTTKRP_Method::names[mttkrp_method] << "." << std::endl;
      }
      if (mttkrp_all_method == MTTKRP_All_Method::Single ||
          mttkrp_all_method == MTTKRP_All_Method::Duplicated) {
        out << "MTTKRP-All method "
            << MTTKRP_All_Method::names[mttkrp_all_method]
            << " is invalid for HIP, changing to ";
        mttkrp_all_method = MTTKRP_All_Method::Atomic;
        out << MTTKRP_All_Method::names[mttkrp_all_method] << "." << std::endl;
      }
      if (method == Solver_Method::GCP_SGD &&
          sampling_type == GCP_Sampling::SemiStratified &&
          fuse && mttkrp_all_method != MTTKRP_All_Method::Atomic) {
        mttkrp_all_method = MTTKRP_All_Method::Atomic;
        out << "Fused semi-stratified sampling/MTTKRP method requires atomic"
            << " on HIP.  Changing MTTKRP-All method to atomic." << std::endl;
      }
      if (method == Solver_Method::CP_OPT &&
          hess_vec_method == Hess_Vec_Method::Full &&
          (hess_vec_tensor_method == Hess_Vec_Tensor_Method::Single ||
           hess_vec_tensor_method == Hess_Vec_Tensor_Method::Duplicated)) {
        out << "hess-vec tensor method " << Hess_Vec_Tensor_Method::names[hess_vec_tensor_method]
            << " is invalid on HIP.  Changing method to atomic." << std::endl;
        hess_vec_tensor_method = Hess_Vec_Tensor_Method::Atomic;
      }
    } else if (space_prop::is_sycl) {
      if (mttkrp_method == MTTKRP_Method::Single ||
          mttkrp_method == MTTKRP_Method::Duplicated) {
        out << "MTTKRP method " << MTTKRP_Method::names[mttkrp_method]
            << " is invalid for SYCL, changing to ";
        mttkrp_method = MTTKRP_Method::Atomic;
        out << MTTKRP_Method::names[mttkrp_method] << "." << std::endl;
      }
      if (mttkrp_all_method == MTTKRP_All_Method::Single ||
          mttkrp_all_method == MTTKRP_All_Method::Duplicated) {
        out << "MTTKRP-All method "
            << MTTKRP_All_Method::names[mttkrp_all_method]
            << " is invalid for SYCL, changing to ";
        mttkrp_all_method = MTTKRP_All_Method::Atomic;
        out << MTTKRP_All_Method::names[mttkrp_all_method] << "." << std::endl;
      }
      if (method == Solver_Method::GCP_SGD &&
          sampling_type == GCP_Sampling::SemiStratified &&
          fuse && mttkrp_all_method != MTTKRP_All_Method::Atomic) {
        mttkrp_all_method = MTTKRP_All_Method::Atomic;
        out << "Fused semi-stratified sampling/MTTKRP method requires atomic"
            << " on SYCL.  Changing MTTKRP-All method to atomic." << std::endl;
      }
      if (method == Solver_Method::CP_OPT &&
          hess_vec_method == Hess_Vec_Method::Full &&
          (hess_vec_tensor_method == Hess_Vec_Tensor_Method::Single ||
           hess_vec_tensor_method == Hess_Vec_Tensor_Method::Duplicated)) {
        out << "hess-vec tensor method " << Hess_Vec_Tensor_Method::names[hess_vec_tensor_method]
            << " is invalid on SYCL.  Changing method to atomic." << std::endl;
        hess_vec_tensor_method = Hess_Vec_Tensor_Method::Atomic;
      }
    }
  }

}
