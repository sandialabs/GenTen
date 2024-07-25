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

#include "json.hpp"

namespace Genten {

// Note, cannot have parantheses in description

static nlohmann::json json_schema = R"(
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "GenTen",
  "type": "object",
  "definitions": {
      "mttkrp": {
          "description": "MTTKRP algorithmic parameters",
          "type": "object",
          "additionalProperties": false,
          "properties": {
              "method": {
                  "description": "MTTKRP algorithm",
                  "enum": ["default", "orig-kokkos", "atomic", "duplicated", "single", "perm", "row-based", "phan"],
                  "default": "default"
              },
              "all-method": {
                  "description": "MTTKRP algorithm for all modes simultaneously",
                  "enum": ["default", "iterated", "atomic", "duplicated", "single"],
                  "default": "default"
              },
              "nnz-tile-size": {
                  "description": "Nonzero tile size for mttkrp algorithm",
                  "type": "integer",
                  "minimum": 1,
                  "default": 128
              },
              "duplicated-tile-size": {
                  "description": "Factor matrix tile size for duplicated mttkrp algorithm",
                  "type": "integer",
                  "minimum": 0,
                  "default": 0
              },
              "duplicated-threshold": {
                  "description": "Theshold for determining when to not use duplicated mttkrp algorithm.  Set to -1.0 to always use duplicated",
                  "type": "number",
                  "minimum": -1.0,
                  "default": -1.0
              },
              "warmup": {
                  "description": "Do an iteration of mttkrp to warmup for generating accurate timing information",
                  "type": "boolean",
                  "default": false
              }
          }
      },
      "goal": {
          "description": "Goal function parameters",
          "type": "object",
          "additionalProperties": false,
          "properties": {
              "method": {
                  "description": "Goal method",
                  "enum": ["none", "python-module", "python-object"],
                  "default": "none"
              },
              "python-module-name": {
                  "description": "Name of python module containing goal function",
                  "type": "string",
                  "default": "__main__"
              },
              "python-object-name": {
                  "description": "Name of python object in python module containing goal function",
                  "type": "string",
                  "default": "goal"
              }
          }
      }
  },
  "additionalProperties": false,
  "required": [
      "solver-method",
      "tensor",
      "k-tensor"
    ],
  "properties": {
      "exec-space": {
          "description": "Kokkos execution space to run on",
          "enum": ["cuda", "hip", "sycl", "openmp", "threads", "serial", "default"],
          "default": "default"
      },
      "proc-grid": {
          "description": "Number of MPI procesors in each dimension",
          "type": "array",
          "items": {
              "type": "integer",
              "minimum": 1
           }
      },
      "solver-method": {
          "description": "Decomposition method",
          "enum": ["cp-als", "cp-opt", "gcp-sgd", "gcp-fed", "gcp-opt"]
      },
      "debug": {
          "description": "Turn on debugging output",
          "type": "boolean",
          "default": false
      },
      "timings": {
          "description": "Print accurate kernel timing info, but may increase total run time by adding fences",
          "type": "boolean",
          "default": false
      },
      "vtune": {
          "description": "Connect to vtune for Intel-based profiling. Aassumes vtune profiling tool, amplxe-cl, is in your path",
          "type": "boolean",
          "default": false
      },
      "tensor": {
          "description": "Description of the input tensor",
          "type": "object",
          "additionalProperties": false,
          "properties": {
              "input-file": {
                  "description": "Path to input tensor file",
                  "type": "string"
              },
              "index-base": {
                  "description": "Starting integer for tensor indices",
                  "type": "integer",
                  "default": 0
              },
              "compressed": {
                  "description": "Read tensor in gzip compressed format",
                  "type": "boolean",
                  "default": false
              },
              "format": {
                  "description": "Whether tensor is sparse or dense",
                  "type": "string",
                  "default": "sparse"
              },
              "file-type": {
                  "description": "File type of tensor file, text or binary",
                  "type": "string",
                  "default": "auto"
              },
              "parallel-read": {
                  "description": "Use MPI-IO to read tensor file",
                  "type": "boolean",
                  "default": true
              },
              "dims": {
                  "description": "Tensor dimensions for binary files without a header",
                  "type": "array",
                  "items": {
                      "type": "integer",
                      "minimum": 1
                  }
              },
              "nnz": {
                  "description": "Number of tensor nonzeros for binary files without a header",
                  "type": "integer",
                  "minimum": 1
              },
              "value-bits": {
                  "description": "Number of bits used to represent tensor entries in sparse or dense binary files without a header",
                  "type": "integer",
                  "minimum": 16
              },
              "sub-bits": {
                  "description": "Number of bits used to represent subscripts in sparse binary files without a header",
                  "type": "array",
                  "items": {
                      "type": "integer",
                      "minimum": 16
                  }
              },
              "output-file": {
                  "description": "Path to output tensor file when generating tensor randomly",
                  "type": "string"
              },
              "rand-nnz": {
                  "description": "Approximate number of random tensor nonzeros",
                  "type": "integer",
                  "minimum": 1,
                  "default": 1000000
              },
              "rand-dims": {
                  "description": "Random tensor dimensions",
                  "type": "array",
                  "items": {
                      "type": "integer",
                      "minimum": 1
                  },
                  "default": [30,40,50]
              }
          }
      },
      "k-tensor": {
          "description": "Description of the computed K-tensor",
          "type": "object",
          "additionalProperties": false,
          "required": [
              "rank"
            ],
          "properties": {
              "rank": {
                  "description": "Rank of factorization to compute",
                  "type": "integer",
                  "minimum": 1
              },
              "output-file": {
                  "description": "Path to output K-tensor file",
                  "type": "string"
              },
              "dense-reconstruction": {
                  "description": "Path to output K-tensor reconstruction as a dense tensor",
                  "type": "string"
              },
              "sparse-reconstruction": {
                  "description": "Path to output K-tensor reconstruction as a sparse tensor",
                  "type": "string"
              },
              "sparse-reconstruction-tolerance": {
                  "description": "Tolerance for distinguishing zeros and nonzeros in sparse reconstruction",
                  "type": "number",
                  "minimum": 0.0,
                  "default": 0.0
              },
              "initial-guess": {
                  "description": "Type of initial guess to use",
                  "enum": ["rand", "file"],
                  "default": "rand"
              },
              "initial-file": {
                  "description": "Path to initial K-tensor file",
                  "type": "string"
              },
              "distributed-guess": {
                  "description": "How to compute parallel-distributed initial guess",
                  "enum": ["serial", "parallel", "parallel-drew"],
                  "default": "serial"
              },
              "seed": {
                  "description": "Seed for random number generator used in initial guess",
                  "type": "integer",
                  "minimum": 1,
                  "default": 12345
              },
              "prng": {
                  "description": "Use parallel random number generator",
                  "type": "boolean",
                  "default": false
              },
              "scale-guess-by-norm-x": {
                  "description": "Scale initial guess by norm of the tensor",
                  "type": "boolean",
                  "default": true
              },
              "dist-method": {
                  "description": "Distributed Ktensor update method",
                  "enum": ["all-reduce", "all-gather", "tpetra"],
                  "default": "all-reduce"
              },
              "optimize-maps": {
                  "description": "Optimize distributed maps to reduce communication",
                  "type": "boolean",
                  "default": false
              },
              "build-maps-on-device": {
                  "description": "Build distributed maps on the device",
                  "type": "boolean",
                  "default": true
              }
          }
      },
      "cp-als": {
          "type": "object",
          "description": "CP-ALS decomposition algorithm",
          "additionalProperties": false,
          "properties": {
              "maxiters": {
                  "description": "maximum iterations to perform",
                  "type": "integer",
                  "minimum": 1,
                  "default": 100
              },
              "tol": {
                  "description": "Stopping tolerance",
                  "type": "number",
                  "minimum": 0.0,
                  "default": 0.004
              },
              "mttkrp": {
                  "$ref": "#/definitions/mttkrp"
              },
              "full-gram": {
                  "description": "Use full Gram matrix formulation which may be faster than the symmetric formulation on some architectures",
                  "type": "boolean"
              },
              "rank-def-solver": {
                  "description": "Use rank-deficient least-squares solver with full-gram formluation for when gram matrix is singular",
                  "type": "boolean",
                  "default": false
              },
              "rcond": {
                  "description": "Truncation parameter for rank-deficient solver",
                  "type": "number",
                  "minimum": 0.0,
                  "default": 1e-8
              },
              "penalty": {
                  "description": "Penalty term for regularization.  Uuseful if gram matrix is singular",
                  "type": "number",
                  "minimum": 0.0,
                  "default": 0.0
              }
          }
      },
      "cp-opt": {
          "type": "object",
          "description": "CP-OPT decomposition algorithm",
          "additionalProperties": false,
          "properties": {
              "maxiters": {
                  "description": "maximum iterations to perform",
                  "type": "integer",
                  "minimum": 1,
                  "default": 100
              },
              "tol": {
                  "description": "Stopping tolerance",
                  "type": "number",
                  "minimum": 0.0,
                  "default": 0.004
              },
              "mttkrp": {
                  "$ref": "#/definitions/mttkrp"
              },
              "method": {
                  "description": "Optimization method",
                  "enum": ["lbfgsb", "rol"],
                  "default": "lbfgsb"
              },
              "lower": {
                  "description": "Lower bound of factorization",
                  "type": "number",
                  "default": -1e300
              },
              "upper": {
                  "description": "Upper bound of factorization",
                  "type": "number",
                  "default": 1e300
              },
              "rol-file": {
                  "description": "Path to ROL optimization settings",
                  "type": "string",
                  "default": ""
              },
              "ftol": {
                  "description": "relative residual reduction tolerance for L-BFGS-B",
                  "type": "number",
                  "minimum": 0.0,
                  "default": 1e-10
              },
              "gtol": {
                  "description": "gradient tolerance for L-BFGS-B",
                  "type": "number",
                  "minimum": 0.0,
                  "default": 1e-5
              },
              "memory": {
                  "description": "memory parameter for L-BFGS-B",
                  "type": "integer",
                  "minimum": 1,
                  "default": 5
              },
              "sub-iters": {
                  "description": "Max number of inner iterations for L-BFGS-B",
                  "type": "integer",
                  "minimum": 0,
                  "default": 10
              },
              "hess-vec": {
                  "description": "Hessian-vector product method",
                  "enum": ["full", "gauss-newton", "finite-difference"],
                  "default": "finite-difference"
              },
              "hess-vec-tensor": {
                  "description": "Hessian-vector product method for tensor-only term",
                  "enum": ["default", "atomic", "duplicated", "single", "perm"],
                  "default": "default"
              },
              "hess-vec-prec": {
                  "description": "Preconditioning method for Hessian-vector product",
                  "enum": ["none", "approx-block-diag"],
                  "default": "approx-block-diag"
              },
              "penalty": {
                  "description": "Penalty term for Tikhonov regularization.  Needed to eliminate scaling ambiguity of CP decomposition.",
                  "type": "number",
                  "minimum": 0.0,
                  "default": 0.0
              }
          }
      },
      "gcp-opt": {
          "type": "object",
          "description": "GCP-OPT decomposition algorithm",
          "additionalProperties": false,
          "properties": {
              "type": {
                  "description": "Loss function type for GCP",
                  "type": "string",
                  "default": "gaussian"
              },
              "eps": {
                  "description": "Perturbation of loss functions for entries near 0",
                  "type": "number",
                  "minimum": 0.0,
                  "maximum": 1.0,
                  "default": 1e-10
              },
              "param": {
                  "description": "Generic parameter for use in some loss functions",
                  "type": "number",
                  "default": 0.0
              },
              "maxiters": {
                  "description": "maximum iterations to perform",
                  "type": "integer",
                  "minimum": 1,
                  "default": 100
              },
              "tol": {
                  "description": "Stopping tolerance",
                  "type": "number",
                  "minimum": 0.0,
                  "default": 0.004
              },
              "mttkrp": {
                  "$ref": "#/definitions/mttkrp"
              },
              "method": {
                  "description": "Optimization method",
                  "enum": ["lbfgsb", "rol"],
                  "default": "lbfgsb"
              },
              "rol-file": {
                  "description": "Path to ROL optimization settings",
                  "type": "string",
                  "default": ""
              },
              "ftol": {
                  "description": "relative residual reduction tolerance for L-BFGS-B",
                  "type": "number",
                  "minimum": 0.0,
                  "default": 1e-10
              },
              "gtol": {
                  "description": "gradient tolerance for L-BFGS-B",
                  "type": "number",
                  "minimum": 0.0,
                  "default": 1e-5
              },
              "memory": {
                  "description": "memory parameter for L-BFGS-B",
                  "type": "integer",
                  "minimum": 1,
                  "default": 5
              },
              "sub-iters": {
                  "description": "Max number of inner iterations for L-BFGS-B",
                  "type": "integer",
                  "minimum": 0,
                  "default": 10
              },
             "fit": {
                  "description": "Compute fit metric",
                  "type": "boolean",
                  "default": false
             },
             "goal": {
                  "$ref": "#/definitions/goal"
             }
          }
      },
      "gcp-sgd": {
          "type": "object",
          "description": "GCP-SGD decomposition algorithm",
          "additionalProperties": false,
          "properties": {
              "maxiters": {
                  "description": "Maximum iterations to perform",
                  "type": "integer",
                  "minimum": 1,
                  "default": 100
              },
              "tol": {
                  "description": "Stopping tolerance",
                  "type": "number",
                  "minimum": 0.0,
                  "default": 0.001
              },
              "mttkrp": {
                  "$ref": "#/definitions/mttkrp"
              },
              "type": {
                  "description": "Loss function type for GCP",
                  "type": "string",
                  "default": "gaussian"
              },
              "eps": {
                  "description": "Perturbation of loss functions for entries near 0",
                  "type": "number",
                  "minimum": 0.0,
                  "maximum": 1.0,
                  "default": 1e-10
              },
              "param": {
                  "description": "Generic parameter for use in some loss functions",
                  "type": "number",
                  "default": 0.0
              },
              "sampling": {
                  "description": "Sampling method",
                  "enum": ["uniform", "stratified", "semi-stratified", "dense"],
                  "default": "stratified"
              },
              "seed": {
                "description": "Seed for random number generator used in sampling",
                "type": "integer",
                "minimum": 1
              },
              "rate": {
                  "description": "Initial step size",
                  "type": "number",
                  "minimum": 0.0,
                  "default": 1e-3
              },
              "decay": {
                  "description": "Rate step size decreases on fails",
                  "type": "number",
                  "minimum": 0.0,
                  "maximum": 1.0,
                  "default": 0.1
              },
              "fails": {
                  "description": "Maximum number of fails",
                  "type": "integer",
                  "minimum": 0,
                  "default": 10
              },
              "epochiters": {
                  "description": "Iterations per epoch",
                  "type": "integer",
                  "minimum": 1,
                  "default": 1000
              },
              "frozeniters": {
                  "description": "Inner iterations with frozen gradient",
                  "type": "integer",
                  "minimum": 1,
                  "default": 100
              },
              "rngiters": {
                  "description": "Iteration loops in parallel RNG",
                  "type": "integer",
                  "minimum": 128,
                  "default": 100
              },
              "fnzs": {
                  "description": "Nonzero samples for f-est",
                  "type": "integer",
                  "minimum": 0,
                  "default": 0
              },
              "fzs": {
                  "description": "Zero samples for f-est",
                  "type": "integer",
                  "minimum": 0,
                  "default": 0
              },
              "gnzs": {
                  "description": "Nonzero samples for gradient",
                  "type": "integer",
                  "minimum": 0,
                  "default": 0
              },
              "gzs": {
                  "description": "Zero samples for gradient",
                  "type": "integer",
                  "minimum": 0,
                  "default": 0
              },
              "oversample": {
                  "description": "Oversample factor for zero sampling",
                  "type": "number",
                  "minimum": 1.0,
                  "default": 1.1
              },
              "bulk-factor": {
                  "description": "factor for bulk zero sampling",
                  "type": "integer",
                  "minimum": 1,
                  "default": 10
              },
              "fnzw": {
                  "description": "Nonzero sample weight for f-est",
                  "type": "number",
                  "minimum": -1.0,
                  "default": -1.0
              },
              "fzw": {
                  "description": "Zero sample weight for f-est",
                  "type": "number",
                  "minimum": -1.0,
                  "default": -1.0
              },
              "gnzw": {
                  "description": "Nonzero sample weight for gradient",
                  "type": "number",
                  "minimum": -1.0,
                  "default": -1.0
              },
              "gzw": {
                  "description": "Zero sample weight for gradient",
                  "type": "number",
                  "minimum": -1.0,
                  "default": -1.0
              },
              "normalize": {
                  "description": "Normalize initial Ktensor",
                  "type": "boolean",
                  "default": true
              },
              "hash": {
                  "description": "Compute hash map for zero sampling",
                  "type": "boolean",
                  "default": false
              },
              "fuse": {
                  "description": "Fuse gradient sampling and MTTKRP",
                  "type": "boolean",
                  "default": false
              },
              "fuse-sa": {
                  "description": "Fuse with sparse array gradient",
                  "type": "boolean",
                  "default": false
              },
              "fit": {
                  "description": "Compute fit metric",
                  "type": "boolean",
                  "default": false
              },
              "step": {
                  "description": "GCP-SGD optimization step type",
                  "enum": ["sgd", "adam", "adagrad", "amsgrad", "sgd-momentum"],
                  "default": "adam"
              },
              "adam-beta1": {
                  "description": "Decay rate for 1st moment average",
                  "type": "number",
                  "minimum": 0.0,
                  "maximum": 1.0,
                  "default": 0.9
              },
              "adam-beta2": {
                  "description": "Decay rate for 2st moment average",
                  "type": "number",
                  "minimum": 0.0,
                  "maximum": 1.0,
                  "default": 0.999
              },
              "adam-eps": {
                  "description": "Shift in ADAM step",
                  "type": "number",
                  "minimum": 0.0,
                  "maximum": 1.0,
                  "default": 1e-8
              },
              "async": {
                  "description": "Asynchronous SGD solver",
                  "type": "boolean",
                  "default": false
              },
              "annealer": {
                  "description": "Step size annealer parameters",
                  "type": "object",
                  "additionalProperties": false,
                  "properties": {
                      "method": {
                          "description": "Annealing method",
                          "enum": ["traditional", "cosine"],
                          "default": "traditional"
                      },
                      "max": {
                          "description": "Initial max learning rate for cosine annealer",
                          "type": "number",
                          "minimum": 0.0,
                          "default": 1e-9
                      },
                      "min": {
                          "description": "Initial min learning rate for cosine annealer",
                          "type": "number",
                          "minimum": 0.0,
                          "default": 1e-12
                      },
                      "temp": {
                          "description": "Initial temperature",
                          "type": "number",
                          "minimum": 0.0,
                          "default": 10.0
                      }
                  }
              },
              "goal": {
                "$ref": "#/definitions/goal"
              }
          }
      },
      "gcp-fed": {
          "type": "object",
          "description": "GCP decomposition algorithm based on federated learning",
          "additionalProperties": false,
          "properties": {
            "maxiters": {
                  "description": "Maximum iterations to perform",
                  "type": "integer",
                  "minimum": 1,
                  "default": 100
              },
              "tol": {
                  "description": "Stopping tolerance",
                  "type": "number",
                  "minimum": 0.0,
                  "default": 0.001
              },
              "mttkrp": {
                  "$ref": "#/definitions/mttkrp"
              },
              "type": {
                  "description": "Loss function type for GCP",
                  "type": "string",
                  "default": "gaussian"
              },
              "eps": {
                  "description": "Perturbation of loss functions for entries near 0",
                  "type": "number",
                  "minimum": 0.0,
                  "maximum": 1.0,
                  "default": 1e-10
              },
              "param": {
                  "description": "Generic parameter for use in some loss functions",
                  "type": "number",
                  "default": 0.0
              },
              "sampling": {
                  "description": "Sampling method",
                  "enum": ["uniform", "stratified", "semi-stratified", "dense"],
                  "default": "stratified"
              },
              "seed": {
                "description": "Seed for random number generator used in sampling",
                "type": "integer",
                "minimum": 1
              },
              "rate": {
                  "description": "Initial step size",
                  "type": "number",
                  "minimum": 0.0,
                  "default": 1e-3
              },
              "decay": {
                  "description": "Rate step size decreases on fails",
                  "type": "number",
                  "minimum": 0.0,
                  "maximum": 1.0,
                  "default": 0.1
              },
              "fails": {
                  "description": "Maximum number of fails",
                  "type": "integer",
                  "minimum": 0,
                  "default": 10
              },
              "epochiters": {
                  "description": "Iterations per epoch",
                  "type": "integer",
                  "minimum": 1,
                  "default": 1000
              },
              "fnzs": {
                  "description": "Nonzero samples for f-est",
                  "type": "integer",
                  "minimum": 0,
                  "default": 0
              },
              "fzs": {
                  "description": "Zero samples for f-est",
                  "type": "integer",
                  "minimum": 0,
                  "default": 0
              },
              "gnzs": {
                  "description": "Nonzero samples for gradient",
                  "type": "integer",
                  "minimum": 0,
                  "default": 0
              },
              "gzs": {
                  "description": "Zero samples for gradient",
                  "type": "integer",
                  "minimum": 0,
                  "default": 0
              },
              "fnzw": {
                  "description": "Nonzero sample weight for f-est",
                  "type": "number",
                  "minimum": -1.0,
                  "default": -1.0
              },
              "fzw": {
                  "description": "Zero sample weight for f-est",
                  "type": "number",
                  "minimum": -1.0,
                  "default": -1.0
              },
              "gnzw": {
                  "description": "Nonzero sample weight for gradient",
                  "type": "number",
                  "minimum": -1.0,
                  "default": -1.0
              },
              "gzw": {
                  "description": "Zero sample weight for gradient",
                  "type": "number",
                  "minimum": -1.0,
                  "default": -1.0
              },
              "normalize": {
                  "description": "Normalize initial Ktensor",
                  "type": "boolean",
                  "default": true
              },
              "hash": {
                  "description": "Compute hash map for zero sampling",
                  "type": "boolean",
                  "default": false
              },
              "fuse": {
                  "description": "Fuse gradient sampling and MTTKRP",
                  "type": "boolean",
                  "default": false
              },
              "fit": {
                  "description": "Compute fit metric",
                  "type": "boolean",
                  "default": false
              },
              "step": {
                  "description": "GCP-SGD optimization step type",
                  "enum": ["sgd", "adam", "adagrad", "amsgrad", "sgd-momentum"],
                  "default": "adam"
              },
              "adam-beta1": {
                  "description": "Decay rate for 1st moment average",
                  "type": "number",
                  "minimum": 0.0,
                  "maximum": 1.0,
                  "default": 0.9
              },
              "adam-beta2": {
                  "description": "Decay rate for 2st moment average",
                  "type": "number",
                  "minimum": 0.0,
                  "maximum": 1.0,
                  "default": 0.999
              },
              "adam-eps": {
                  "description": "Shift in ADAM step",
                  "type": "number",
                  "minimum": 0.0,
                  "maximum": 1.0,
                  "default": 1e-8
              },

              "method": {
                  "description": "Federated learning method",
                  "enum": ["fedopt", "fedavg"],
                  "default": "fedopt"
              },
              "meta-step": {
                  "description": "Meta optimization step type",
                  "enum": ["sgd", "adam", "adagrad", "amsgrad", "sgd-momentum", "demon"],
                  "default": "adam"
              },
              "meta-rate": {
                  "description": "Overall learning rate",
                  "type": "number",
                  "minimum": 0.0,
                  "default": 1e-3
              },
              "annealer": {
                  "description": "Step size annealer parameters",
                  "type": "object",
                  "additionalProperties": false,
                  "properties": {
                      "method": {
                          "description": "Annealing method",
                          "enum": ["traditional", "cosine"],
                          "default": "traditional"
                      },
                      "max": {
                          "description": "Initial max learning rate for cosine annealer",
                          "type": "number",
                          "minimum": 0.0,
                          "default": 1e-9
                      },
                      "min": {
                          "description": "Initial min learning rate for cosine annealer",
                          "type": "number",
                          "minimum": 0.0,
                          "default": 1e-12
                      },
                      "temp": {
                          "description": "Initial temperature",
                          "type": "number",
                          "minimum": 0.0,
                          "default": 10.0
                      }
                  }
              },
              "downpour-iters": {
                  "description": "Number of downpour iterations",
                  "type": "integer",
                  "minimum": 1,
                  "default": 4
              }
          }
      },
      "streaming-gcp": {
          "type": "object",
          "description": "Streaming GCP decomposition algorithm",
          "additionalProperties": false,
          "properties": {
              "solver": {
                  "description": "Streaming solver algorithm",
                  "enum": ["sgd", "least-squares", "online-cp"],
                  "default": "sgd"
              },
              "history-method": {
                  "description": "History term formulation",
                  "enum": ["ktensor-fro", "factor-fro", "gcp-loss"],
                  "default": "ktensor-fro"
              },
              "window-method": {
                  "description": "Algorithm for choosing samples in the window",
                  "enum": ["reservoir", "last"],
                  "default": "reservoir"
              },
              "window-size": {
                  "description": "Number of slices in streaming history window",
                  "type": "integer",
                  "minimum": 0,
                  "default": 0
              },
              "window-weight": {
                  "description": "Multiplier for each streaming window term",
                  "type": "number",
                  "minimum": 0.0,
                  "default": 0.0
              },
              "window-penalty": {
                  "description": "Multiplier for entire streaming window",
                  "type": "number",
                  "minimum": 0.0,
                  "default": 0.0
              },
              "factor-penalty": {
                  "description": "Penalty term on factor matrices",
                  "type": "number",
                  "minimum": 0.0,
                  "default": 0.0
              }
          }
      },
      "ttm": {
          "description": "TTM algorithmic parameters",
          "type": "object",
          "additionalProperties": false,
          "properties": {
              "method": {
                  "description": "Method for TTM implementation",
                  "enum": ["dgemm", "parfor-dgemm"],
                  "default": "dgemm"
              }
          }
      },
      "testing": {
          "description": "Regression testing parameters",
          "type": "object",
          "additionalProperties": false,
          "properties": {
              "final-fit": {
                  "description": "Tests on the final fit",
                  "type": "object",
                  "additionalProperties": false,
                  "properties": {
                      "value" : {
                          "type": "number",
                          "minimum": 0.0,
                          "maximum": 1.0
                      },
                      "relative-tolerance" : {
                          "type": "number",
                          "minimum": 0.0,
                          "default": 0.0
                      },
                      "absolute-tolerance" : {
                          "type": "number",
                          "minimum": 0.0,
                          "default": 0.0
                      }
                  }
              },
              "final-residual": {
                  "description": "Tests on the final residual",
                  "type": "object",
                  "additionalProperties": false,
                  "properties": {
                      "value" : {
                          "type": "number",
                          "minimum": 0.0
                      },
                      "relative-tolerance" : {
                          "type": "number",
                          "minimum": 0.0,
                          "default": 0.0
                      },
                      "absolute-tolerance" : {
                          "type": "number",
                          "minimum": 0.0,
                          "default": 0.0
                      }
                  }
              },
              "final-gradient-norm": {
                  "description": "Tests on the final gradient norm",
                  "type": "object",
                  "additionalProperties": false,
                  "properties": {
                      "value" : {
                          "type": "number",
                          "minimum": 0.0
                      },
                      "relative-tolerance" : {
                          "type": "number",
                          "minimum": 0.0,
                          "default": 0.0
                      },
                      "absolute-tolerance" : {
                          "type": "number",
                          "default": 0.0,
                          "minimum": 0.0
                      }
                  }
              },
              "iterations": {
                  "description": "Tests on the number of iterations",
                  "type": "object",
                  "additionalProperties": false,
                  "properties": {
                      "value" : {
                          "type": "integer",
                          "minimum": 0
                      },
                      "relative-tolerance" : {
                          "type": "integer",
                          "minimum": 0,
                          "default": 0
                      },
                      "absolute-tolerance" : {
                          "type": "integer",
                          "minimum": 0,
                          "default": 0
                      }
                  }
              }
          }
      }
  }
}
)"_json;

}
