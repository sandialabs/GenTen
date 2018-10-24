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


#include <string>
#include <iostream>
#include <sstream>

using namespace std;

#include "Genten_Util.hpp"
#include "Genten_IndxArray.hpp"


string IntToStr( int i ) { std::ostringstream r; r << i; return r.str(); }

ttb_real parse_ttb_real(int argc, char** argv, string cl_arg, ttb_real default_value, ttb_real min=0.0, ttb_real max=1.0) {
  int arg=1;
  char *cend = 0;
  ttb_real tmp;
  while (arg < argc) {
    if (cl_arg == string(argv[arg])) {
      // get next cl_arg
      arg++;
      if (arg >= argc)
        return default_value;
      // convert to ttb_real
      tmp = strtod(argv[arg],&cend);
      // check if cl_arg is actuall a ttb_real
      if (argv[arg] == cend) {
        std::ostringstream error_string;
        error_string << "Unparseable input: " << cl_arg << " " << argv[arg] << ", must be a double" << endl;
        Genten::error(error_string.str());
        exit(1);
        // check if ttb_real is within bounds
      }
      if (tmp < min || tmp > max) {
        std::ostringstream error_string;
        error_string << "Bad input: " << cl_arg << " " << argv[arg] << ",  must be in the range (" << min << ", " << max << ")" << endl;
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

ttb_indx parse_ttb_indx(int argc, char** argv, string cl_arg, ttb_indx default_value, ttb_indx min=0, ttb_indx max=100) {
  int arg=1;
  char *cend = 0;
  ttb_indx tmp;
  while (arg < argc) {
    if (cl_arg == string(argv[arg])) {
      // get next cl_arg
      arg++;
      if (arg >= argc)
        return default_value;
      // convert to ttb_real
      tmp = strtol(argv[arg],&cend,10);
      // check if cl_arg is actuall a ttb_real
      if (argv[arg] == cend) {
        std::ostringstream error_string;
        error_string << "Unparseable input: " << cl_arg << " " << argv[arg] << ", must be an unsigned integer" << endl;
        Genten::error(error_string.str());
        exit(1);
        // check if ttb_real is within bounds
      }
      if (tmp < min || tmp > max) {
        std::ostringstream error_string;
        error_string << "Bad input: " << cl_arg << " " << argv[arg] << ",  must be in the range (" << min << ", " << max << ")" << endl;
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

ttb_bool parse_ttb_bool(int argc, char** argv, string cl_arg, ttb_bool default_value) {
  int arg=1;
  while (arg < argc) {
    if (cl_arg == string(argv[arg])) {
      // return true if arg is found
      return true;
    }
    arg++;
  }
  // return default value if not specified on command line
  return default_value;
}

string parse_string(int argc, char** argv, string cl_arg, string default_value) {
  int arg=1;
  string tmp;
  while (arg < argc) {
    if (cl_arg == string(argv[arg])) {
      // get next cl_arg
      arg++;
      if (arg >= argc)
        return "";
      // convert to string
      tmp = string(argv[arg]);
      // return ttb_real if everything is OK
      return tmp;
    }
    arg++;
  }
  // return default value if not specified on command line
  return default_value;

}

Genten::IndxArray parse_ttb_indx_array(int argc, char** argv, string cl_arg, const Genten::IndxArray& default_value, ttb_indx min=1, ttb_indx max=INT_MAX) {
  int arg=1;
  char *cend = 0;
  ttb_indx tmp;
  std::vector<ttb_indx> vals;
  while (arg < argc) {
    if (cl_arg == string(argv[arg])) {
      // get next cl_arg
      arg++;
      if (arg >= argc)
        return default_value;
      char *arg_val = argv[arg];
      if (arg_val[0] != '[') {
        std::ostringstream error_string;
        error_string << "Unparseable input: " << cl_arg << " " << arg_val << ", must be of the form { int, ... }" << endl;
        Genten::error(error_string.str());
        exit(1);
      }
      while (strlen(arg_val) > 0 && arg_val[0] != ']') {
        ++arg_val; // Move past ,
        // convert to ttb_real
        tmp = strtol(arg_val,&cend,10);
        // check if cl_arg is actuall a ttb_real
        if (arg_val == cend) {
          std::ostringstream error_string;
          error_string << "Unparseable input: " << cl_arg << " " << arg_val << ", must be of the form { int, ... }" << endl;
          Genten::error(error_string.str());
          exit(1);
        }
        // check if ttb_indx is within bounds
        if (tmp < min || tmp > max) {
          std::ostringstream error_string;
          error_string << "Bad input: " << cl_arg << " " << arg_val << ",  must be in the range (" << min << ", " << max << ")" << endl;
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

template <typename T>
T parse_ttb_enum(int argc, char** argv, string cl_arg, T default_value,
                 unsigned num_values, const T* values, const char*const* names) {
  int arg=1;
  while (arg < argc) {
    if (cl_arg == string(argv[arg])) {
      // get next cl_arg
      arg++;
      if (arg >= argc)
        return default_value;
      // convert to string
      string arg_val = string(argv[arg]);
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
      error_string << "." << endl;
      Genten::error(error_string.str());
      exit(1);
    }
    arg++;
  }
  // return default value if not specified on command line
  return default_value;

}
