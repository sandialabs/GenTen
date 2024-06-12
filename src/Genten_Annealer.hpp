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

#include <cmath>
#include <memory>

#include "Genten_AlgParams.hpp"

namespace Genten {
class AnnealerBase {
public:
  AnnealerBase(const AlgParams&) {}
  virtual ~AnnealerBase() = default;
  virtual ttb_real operator()(int epoch) = 0;
  virtual void failed(){};
  virtual void success(){};
  virtual void print(std::ostream& os) {};

private:
};

class TraditionalAnnealer : public AnnealerBase {
  ttb_real step_size_;
  ttb_real decay_;

public:
  TraditionalAnnealer(const AlgParams& algParams)
    : AnnealerBase(algParams),
      step_size_(algParams.rate),
      decay_(algParams.decay) {}

  ttb_real operator()(int epoch) override { return step_size_; }
  void failed() override { step_size_ *= decay_; }
  void print(std::ostream& os) override {
    os << "Traditional annealer, learning rate: "
       << std::setprecision(1) << std::scientific << step_size_
       << ", decay: " << decay_ << std::endl;
  }
};

// Slightly annoyingly I want the first run through to be a warm up then resets
// So we will maintain a different count than Tcur to decide when to reset Tcur
class CosineAnnealer : public AnnealerBase {
  ttb_real min_lr;
  ttb_real max_lr;
  int Ti;
  int Tcur;
  int iter = 0;

public:
  CosineAnnealer(const AlgParams& algParams)
    : AnnealerBase(algParams),
      min_lr(algParams.anneal_min_lr),
      max_lr(algParams.anneal_max_lr),
      Ti(algParams.anneal_Ti),
      Tcur(Ti) {}

  ttb_real operator()(int) override {
    return min_lr +
           0.5 * (max_lr - min_lr) * (1 + std::cos(ttb_real(Tcur) / Ti * M_PI));
  }

  void failed() override {
    min_lr *= 0.5;
    max_lr *= 0.5;
    Tcur = 0;
    iter = 0;
  }

  void success() override {
    ++iter;
    ++Tcur;
    if (iter > Ti) {
      Tcur = 0;
      iter = 0;
      Ti *= 2;
    }
  }

  void print(std::ostream& os) override {
    os << "Cosine annealer, min learning rate: "
       << std::setprecision(1) << std::scientific << min_lr
       << ", max learning rate: "
       << std::setprecision(1) << std::scientific << max_lr
       << ", initial temp: "
       << std::setprecision(1) << std::scientific << Ti
       << std::endl;
  }
};

inline std::unique_ptr<AnnealerBase> getAnnealer(const AlgParams& algParams){
  if (algParams.annealer == GCP_AnnealerMethod::Traditional)
    return std::make_unique<TraditionalAnnealer>(TraditionalAnnealer(algParams));
  else if(algParams.annealer == GCP_AnnealerMethod::Cosine)
    return std::make_unique<CosineAnnealer>(CosineAnnealer(algParams));
  else
    return std::make_unique<TraditionalAnnealer>(TraditionalAnnealer(algParams));
}

} // namespace Genten
