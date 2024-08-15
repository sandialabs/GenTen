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

#include <string>

#include "Genten_Tensor.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_AlgParams.hpp"

#ifdef HAVE_PYTHON_EMBED
#include <pybind11/embed.h>
#endif

namespace Genten {

  // Abstract class representing a goal function
  template <typename ExecSpace>
  class GCP_Goal {
  public:

    using exec_space = ExecSpace;
    using ktensor_type = KtensorT<exec_space>;

    GCP_Goal() = default;

    virtual ~GCP_Goal() {}

    // Compute information that is common to both value() and gradient()
    // when a new design vector is computed
    virtual void update(const ktensor_type& M) = 0;

    // Compute value of the objective function:
    virtual ttb_real value(const ktensor_type& M) const = 0;

    // Compute gradient of objective function:
    virtual void gradient(ktensor_type& G, const ktensor_type& M,
                          const ttb_real s) const = 0;
  };

#ifdef HAVE_PYTHON_EMBED
  // A goal function implemented as a python object
  template <typename ExecSpace>
  class __attribute__((visibility("default"))) GCP_PythonObjectGoal :
    public GCP_Goal<ExecSpace> {
  public:

    using exec_space = ExecSpace;
    using base_type = GCP_Goal<exec_space>;
    using ktensor_type = typename base_type::ktensor_type;
    using tensor_type = TensorT<exec_space>;

    GCP_PythonObjectGoal(const tensor_type& X,
                         const ktensor_type& M,
                         const pybind11::object& po) : python_object(po) {}

    virtual ~GCP_PythonObjectGoal() {}

    // Compute information that is common to both value() and gradient()
    // when a new design vector is computed
    virtual void update(const ktensor_type& M) override
    {
      Ktensor Mh = create_mirror_view(M);
      deep_copy(Mh, M);
      try {
        python_object.attr("update")(Mh);
      }
      catch (pybind11::error_already_set& e) {
        Genten::error(e.what());
      }
    }

    // Compute value of the objective function:
    virtual ttb_real value(const ktensor_type& M) const override
    {
      Ktensor Mh = create_mirror_view(M);
      deep_copy(Mh, M);
      ttb_real res = 0.0;
      try {
        res = python_object.attr("value")(Mh).template cast<ttb_real>();
      }
      catch (pybind11::error_already_set& e) {
        Genten::error(e.what());
      }
      return res;
    }

    // Compute gradient of objective function:
    virtual void gradient(ktensor_type& G, const ktensor_type& M,
                          const ttb_real s) const override
    {
      Ktensor Mh = create_mirror_view(M);
      deep_copy(Mh, M);
      try {
        Ktensor Hh =
          python_object.attr("gradient")(Mh).template cast<Ktensor>();
        ktensor_type H = create_mirror_view(exec_space(), Hh);
        deep_copy(H, Hh);
        const ttb_indx nd = G.ndims();
        for (ttb_indx n=0; n<nd; ++n)
          G[n].plus(H[n],s);
      }
      catch (pybind11::error_already_set& e) {
        Genten::error(e.what());
      }
    }

  protected:
    pybind11::object python_object;
  };
#endif

  template <typename ExecSpace>
  GCP_Goal<ExecSpace>*
  goalFactory(const TensorT<ExecSpace>& X,
              const KtensorT<ExecSpace>& M,
              const AlgParams& algParams)
  {
    GCP_Goal<ExecSpace>* goal = nullptr;
    if (algParams.goal_method == GCP_Goal_Method::None)
      goal = nullptr;
    else if (algParams.goal_method == GCP_Goal_Method::PythonObject) {
#ifdef HAVE_PYTHON_EMBED
      goal = new GCP_PythonObjectGoal<ExecSpace>(X, M, algParams.python_object);
#else
      Genten::error("Python object goal requires embedded python (configure with ENABLE_PYTHON_EMBED=ON)!");
#endif
    }
    else if (algParams.goal_method == GCP_Goal_Method::PythonModule) {
#ifdef HAVE_PYTHON_EMBED
      try {
        auto python_module =
          pybind11::module_::import(algParams.python_module_name.c_str());
        auto python_object =
          python_module.attr(algParams.python_object_name.c_str());
        if (pybind11::hasattr(python_object, "init")) {
          Tensor Xh = create_mirror_view(X);
          Ktensor Mh = create_mirror_view(M);
          deep_copy(Xh, X);
          deep_copy(Mh, M);
          python_object.attr("init")(Xh, Mh);
        }
        goal = new GCP_PythonObjectGoal<ExecSpace>(X, M, python_object);
      }
      catch (pybind11::error_already_set& e) {
        Genten::error(e.what());
      }
#else
      Genten::error("Python module goal requires embedded python (configure with ENABLE_PYTHON_EMBED=ON)!");
#endif
    }
    else
      Genten::error("Unknown goal");
    return goal;
  }

}
