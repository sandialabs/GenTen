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

#include "Genten_Ptree.hpp"

namespace Genten {
class AnnealerBase {
public:
  AnnealerBase(ptree const &ptree) {}
  virtual ~AnnealerBase() = default;
  virtual double operator()(int epoch) = 0;
  virtual void failed(){};
  virtual void success(){};

private:
};

class TraditionalAnnealer : public AnnealerBase {
  double step_size_;

public:
  TraditionalAnnealer(ptree const &ptree)
    : AnnealerBase(ptree), step_size_(ptree.get_child("learning-rate").get<double>("step", 3e-4)) {}

  double operator()(int epoch) override { return step_size_; }
  void failed() override { step_size_ /= 10; }
};

// Slightly annoyingly I want the first run through to be a warm up then resets
// So we will maintain a different count than Tcur to decide when to reset Tcur
class CosineAnnealer : public AnnealerBase {
  double min_lr;
  double max_lr;
  int Ti;
  int Tcur;
  int iter = 0;

public:
  CosineAnnealer(ptree const &ptree)
    : AnnealerBase(ptree), min_lr(ptree.get_child("learning-rate").get<double>("min", 1e-12)),
      max_lr(ptree.get_child("learning-rate").get<double>("max", 1e-9)),
      Ti(ptree.get_child("learning-rate").get<int>("Ti", 10)), Tcur(Ti) {}

  double operator()(int) override {
    return min_lr +
           0.5 * (max_lr - min_lr) * (1 + std::cos(double(Tcur) / Ti * M_PI));
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
};

inline std::unique_ptr<AnnealerBase> getAnnealer(ptree const& ptree){
  auto annealer = ptree.get<std::string>("annealer", "traditional");
  if(annealer == "traditional"){
    return std::make_unique<TraditionalAnnealer>(TraditionalAnnealer(ptree));
  } else if(annealer == "cosine"){
    return std::make_unique<CosineAnnealer>(CosineAnnealer(ptree));
  } else {
    return std::make_unique<TraditionalAnnealer>(TraditionalAnnealer(ptree));
  }
}

} // namespace Genten
