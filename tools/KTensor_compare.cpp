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

#include "Genten_Boost.hpp"
#include "Genten_IOtext.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_MixedFormatOps.hpp"
#include "Genten_Sptensor.hpp"

#include <boost/property_tree/json_parser.hpp>

#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using Exec = Kokkos::DefaultHostExecutionSpace;
using KTensor = Genten::KtensorT<Exec>;
using NamedKT = std::pair<std::string, KTensor>;
using STensor = Genten::SptensorT<Exec>;

std::vector<std::string> KTensorFiles(Genten::ptree const &in);

std::vector<NamedKT> readKtensors(std::vector<std::string> const &files);

STensor readSpTensor(std::string const &file_name, int index_base);

double computeError(STensor const &ten, NamedKT const &nkt);
void computeCosineSimOfFactors(NamedKT const &a, NamedKT const &b);

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "k-comp takes one argument an input json file.\n";
    return -1;
  }
  Kokkos::initialize(argc, argv);
  {

    Genten::ptree input;
    boost::property_tree::read_json(argv[1], input);

    auto sp_tensor_file = input.get_optional<std::string>("tensor_file");

    auto ktensor_files = KTensorFiles(input);

    boost::optional<STensor> stensor;
    if (sp_tensor_file) {
      std::cout << "Sparse tensor file: " << sp_tensor_file.get() << "\n";
      auto index_base = input.get<int>("index_base", 0);
      stensor = readSpTensor(sp_tensor_file.get(), index_base);
    }

    std::cout << "KTensor files to compare:\n";
    for (auto const &file : ktensor_files) {
      std::cout << "\t" << file << "\n";
    }

    auto ktensors = readKtensors(ktensor_files);

    for (auto ktensor : ktensors) {
      auto fit = computeError(stensor.get(), ktensor);
      std::cout << "Fit for " << ktensor.first << ": " << std::setprecision(10)
                << fit << "\n";
    }

    computeCosineSimOfFactors(ktensors[0], ktensors[1]);
  }
  Kokkos::finalize();
  return 0;
}

std::vector<std::string> KTensorFiles(Genten::ptree const &in) {
  std::vector<std::string> files;
  for (auto const &file : in.get_child("k_files")) {
    files.push_back(file.second.data());
  }

  return files;
}

std::vector<NamedKT> readKtensors(std::vector<std::string> const &files) {
  std::vector<NamedKT> out;

  for (const auto &file : files) {
    KTensor kt;
    Genten::import_ktensor(file, kt);
    out.push_back(std::make_pair(file, std::move(kt)));
  }

  return out;
}

STensor readSpTensor(std::string const &file_name, int index_base) {
  STensor out;
  Genten::import_sptensor(file_name, out, index_base);
  return out;
}

double computeError(STensor const &ten, NamedKT const &nkt) {
  static double ten_norm = ten.norm();
  double ten_norm2 = ten_norm * ten_norm;
  double knorm2 = nkt.second.normFsq();
  double dot = Genten::innerprod(ten, nkt.second);

  return std::sqrt(ten_norm2 + knorm2 - 2.0 * dot) / ten_norm;
}

void computeCosineSimOfFactors(NamedKT const &a, NamedKT const &b) {
  KTensor const &ak = a.second;
  KTensor const &bk = b.second;

  const auto nfactors = ak.ndims();
  const auto rank = ak.ncomponents();

  for (auto i = 0; i < nfactors; ++i) {
    Genten::FacMatrixT<Exec> const &fa = ak.factors()[i];
    Genten::FacMatrixT<Exec> const &fb = bk.factors()[i];

    auto const ncols = fa.nCols();
    auto const nrows = fa.nRows();

    using KokkosViewMapType = Eigen::Map<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
        0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

    KokkosViewMapType mapa(fa.view().data(), nrows, ncols,
                           Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                               fa.view().stride_0(), fa.view().stride_1()));

    KokkosViewMapType mapb(fb.view().data(), nrows, ncols,
                           Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                               fb.view().stride_0(), fb.view().stride_1()));

    // Sanity Check
    for (auto i = 0; i < nrows; ++i) {
      for (auto j = 0; j < ncols; ++j) {
        if ((mapa(i, j) != fa.entry(i, j)) || (mapb(i, j) != fb.entry(i, j))) {
          std::terminate();
        }
      }
    }

    Eigen::MatrixXd dist = Eigen::MatrixXd::Zero(ncols, ncols);
    // Cosine distance for each compnent
    for (auto i = 0; i < ncols; ++i) {
      for (auto j = i; j < ncols; ++j) {
        auto cola = mapa.col(i);
        auto colb = mapb.col(j);
        auto norma = cola.norm();
        auto normb = colb.norm();
        auto val = 1.0 - cola.dot(colb)/(norma * normb);
        dist(i,j) = val;
        dist(j,i) = val;
      }
    }
    std::cout << dist.format(Eigen::IOFormat(Eigen::FullPrecision,0,",", "\n","[","]")) << std::endl;

    break;
  }
}
