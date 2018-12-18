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

#include "Genten_Util.hpp"
#include "Genten_IndxArray.hpp"

namespace Genten {

  // Struct for passing various algorithmic parameters
  struct AlgParams {

    // Generic options
    std::string method;  // Solver method ("cp-als", "gcp-opt", "gcp-sgd", ...)
    ttb_indx rank;       // Rank of decomposition
    unsigned long seed;  // Random number seed
    bool prng;           // Use parallel random number generator
    ttb_indx maxiters;   // Maximum number of iterations
    ttb_real maxsecs;    // Maximum amount of time
    ttb_real tol;        // Decomposition tolerance
    ttb_indx printitn;   // Print iterations
    bool debug;          // Print debugging info

    // MTTKRP options
    MTTKRP_Method::type mttkrp_method; // MTTKRP algorithm
    unsigned mttkrp_duplicated_factor_matrix_tile_size; // Tile size for MTTKRP
    bool warmup; // Warmup by calling MTTKRP before decompsition

    // GCP options
    GCP_LossFunction::type loss_function_type; // Loss function for GCP
    ttb_real loss_eps;                         // Perturbation for GCP

    // GCP-Opt options
    std::string rolfilename; // Filename for ROL solver options

    // GCP-SGD options
    ttb_real rate;                       // Initial step size
    ttb_real decay;                      // Rate step size decreases on fails
    ttb_indx max_fails;                  // Maximum number of fails
    ttb_indx epoch_iters;                // Number of iterations per epoch
    ttb_indx num_samples_nonzeros_value; // Nonzero samples for f-est
    ttb_indx num_samples_zeros_value;    // Zero sampels for f-est
    ttb_indx num_samples_nonzeros_grad;  // Nonzero samples for gradient
    ttb_indx num_samples_zeros_grad;     // Zero samples for gradient

    // Constructor initializing values to defaults
    AlgParams();

    // Parse options
    void parse(int argc, char* argv[]);

    // Print help string
    static void print_help(std::ostream& out);

    // Print parmeters
    void print(std::ostream& out);
  };

  ttb_real parse_ttb_real(int argc, char** argv, std::string cl_arg, ttb_real default_value, ttb_real min=0.0, ttb_real max=1.0);
  ttb_indx parse_ttb_indx(int argc, char** argv, std::string cl_arg, ttb_indx default_value, ttb_indx min=0, ttb_indx max=100);
  ttb_bool parse_ttb_bool(int argc, char** argv, std::string cl_arg, ttb_bool default_value);
  std::string parse_string(int argc, char** argv, std::string cl_arg, std::string default_value);
  IndxArray parse_ttb_indx_array(int argc, char** argv, std::string cl_arg, const IndxArray& default_value, ttb_indx min=1, ttb_indx max=INT_MAX);
  template <typename T>
  T parse_ttb_enum(int argc, char** argv, std::string cl_arg, T default_value,
                   unsigned num_values, const T* values,
                   const char*const* names)
  {
    int arg=1;
    while (arg < argc) {
      if (cl_arg == std::string(argv[arg])) {
        // get next cl_arg
        arg++;
        if (arg >= argc)
          return default_value;
        // convert to string
        std::string arg_val = std::string(argv[arg]);
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
      arg++;
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

}
