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

#include "Genten_Boost.hpp"

#include <cmath>

namespace Genten {

class CosineAnnealer {
  double last_returned = 0.0;
  double min_lr;
  double max_lr;
  double warmup_scale;
  int epoch_internal = 0;
  int cycle_size;
  int warmup_size;
  bool do_warmup;

public:
  CosineAnnealer(ptree const &ptree)
      : min_lr(ptree.get<double>("lr.min_lr", 1e-12)),
        max_lr(ptree.get<double>("lr.max_lr", 1e-9)),
        warmup_size(ptree.get<int>("lr.warmup_size", 20)),
        cycle_size(ptree.get<int>("lr.cycle_size", 50)),
        do_warmup(ptree.get<bool>("lr.warmup", true)) {
    if (auto ws = ptree.get_optional<double>("lr.warmup_scale")) {
      warmup_scale = ws.get();
    } else {
      const auto term = std::log(max_lr / min_lr) / warmup_size;
      warmup_scale = std::exp(term);
    }
  }

  inline double operator()(int epoch) {
    if (do_warmup) {
      last_returned = min_lr * std::pow(warmup_scale, epoch_internal);
    } else {
      if (epoch_internal > cycle_size) {
        epoch_internal = 0;
      }

      last_returned =
          min_lr +
          0.5 * (max_lr - min_lr) *
              (1 + std::cos(double(epoch_internal) / cycle_size * M_PI));
    }
    ++epoch_internal;
    if (do_warmup && epoch_internal == warmup_size) {
      epoch_internal = 0; // Start over
      do_warmup = false;
    }

    return last_returned;
  }

  inline void failed() {
    if (do_warmup) {
      max_lr = min_lr * std::pow(warmup_scale, epoch_internal - 2);
      do_warmup = false;
    } else {
      max_lr = 0.5 * last_returned;
    }
    epoch_internal = 0;
  }

  inline void success() {}
};

class BoringAnnealer {
  double min_lr;
  double max_lr;
  double warmup_scale;
  int warmup_size;
  bool do_warmup;
  int epoch_internal = 0;

public:
  BoringAnnealer(ptree const &ptree)
      : min_lr(ptree.get<double>("lr.min_lr", 1e-12)),
        max_lr(ptree.get<double>("lr.max_lr", 1e-9)),
        warmup_size(ptree.get<int>("lr.warmup_size", 20)),
        do_warmup(ptree.get<bool>("lr.warmup", true)) {
    if (auto ws = ptree.get_optional<double>("lr.warmup_scale")) {
      warmup_scale = ws.get();
    } else {
      const auto term = std::log(max_lr / min_lr) / warmup_size;
      warmup_scale = std::exp(term);
    }
  }

  inline double operator()(int epoch) {
    auto out = 0.0;
    if (do_warmup) {
      out = min_lr * std::pow(warmup_scale, epoch_internal);
    } else {
      out = max_lr;
    }
    ++epoch_internal;
    if (epoch_internal == warmup_size) {
      epoch_internal = 0; // Start over
      do_warmup = false;
    }

    return out;
  }

  inline void failed() {
    if (do_warmup) {
      max_lr = min_lr * std::pow(warmup_scale, epoch_internal - 2);
      do_warmup = false;
    } else {
      // Half the distance between max and min_lr
      max_lr = (max_lr - min_lr)/2.0 + min_lr;
    }
    epoch_internal = 0;
  }

  inline void success() {}
};

} // namespace Genten
