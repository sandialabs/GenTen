//@HEADER
//// ************************************************************************
////     Genten: Software for Generalized Tensor Decompositions
////     by Sandia National Laboratories
////
//// Sandia National Laboratories is a multimission laboratory managed
//// and operated by National Technology and Engineering Solutions of Sandia,
//// LLC, a wholly owned subsidiary of Honeywell International, Inc., for the
//// U.S. Department of Energy's National Nuclear Security Administration under
//// contract DE-NA0003525.
////
//// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
//// (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
//// Government retains certain rights in this software.
////
//// Redistribution and use in source and binary forms, with or without
//// modification, are permitted provided that the following conditions are
//// met:
////
//// 1. Redistributions of source code must retain the above copyright
//// notice, this list of conditions and the following disclaimer.
////
//// 2. Redistributions in binary form must reproduce the above copyright
//// notice, this list of conditions and the following disclaimer in the
//// documentation and/or other materials provided with the distribution.
////
//// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
//// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
//// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//// ************************************************************************
////@HEADER
//
#pragma once
//
//HK#include<Kokkos_Core.hpp>
//HK#include<KokkosBlas3_gemm.hpp>
//HK#include <KokkosBatched_Gemm_Decl.hpp>
//HK#include <KokkosBatched_Gemm_Team_Impl.hpp>

//HKusing namespace KokkosBatched;

template <typename ExecSpace>
struct ComputeCokurtElems{
  typedef typename Kokkos::View<double*, ExecSpace> sub_view_type;
  typedef typename sub_view_type::device_type device_type;
  typedef typename Kokkos::TeamPolicy<ExecSpace>::member_type member_type;
  typedef typename Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace> raw_view_type;
  typedef typename Kokkos::View<double**, Kokkos::LayoutLeft, ExecSpace> sub_matrix_view_type;

  int nrows_;
  int start_col_;
  int nsamples_;
  int l_;
  sub_matrix_view_type local_sub_matrix_;
  raw_view_type input_raw_data_;

  ComputeCokurtElems(const raw_view_type& inp_raw_data, sub_matrix_view_type& sub_matrix, int nrows, int start_col, int l,
                     int nsamples):
          input_raw_data_(inp_raw_data), local_sub_matrix_(sub_matrix),
          nrows_(nrows), start_col_(start_col), l_(l), nsamples_(nsamples)
  {}
  KOKKOS_INLINE_FUNCTION void
  operator() (const member_type team_member) const{
    //HKtypedef Mode::Team mode_type;
    //HKtypedef Algo::Level3::Unblocked algo_type;

    //Figure out the "j" and "k" indices based on team rank 
    int local_col = team_member.league_rank();
    int jk  = local_col + start_col_; // team_member.league_rank() + start_col_;
    int k = jk / nrows_;
    int j = jk % nrows_; 

    //HK if(team_member.team_rank()==0){printf("Team %d j/k/l are: %d/%d/%d\n",team_member.league_rank(),j,k,l_);}

    // int nthreads = team_member.team_size();  // unused

    auto l_vec = Kokkos::subview(input_raw_data_, l_, Kokkos::ALL);
    auto k_vec = Kokkos::subview(input_raw_data_, k,  Kokkos::ALL);
    auto j_vec = Kokkos::subview(input_raw_data_, j,  Kokkos::ALL);

    //Add code here to compute each element of the submatrix
    //submatrix is a slice of the cokurtosis tensor
      for(int i=0; i<nrows_; i++){
        auto i_vec = Kokkos::subview(input_raw_data_, i, Kokkos::ALL);

        double e_ijkl=0.0, e_ij=0.0, e_kl=0.0, e_ik=0.0, e_jl = 0.0, e_il = 0.0, e_jk = 0.0; 
        
        //e_ijkl
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team_member, nsamples_),
          [=] (int col, double & innerUpdate) { 
             innerUpdate += i_vec(col) * j_vec(col) * k_vec(col) * l_vec(col); 
          }, e_ijkl);

        e_ijkl /= nsamples_;
         
        //IF Only Raw (not centered) moment is desired
        //local_sub_matrix_(i,local_col) = e_ijkl;
        
        //e_ij
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team_member, nsamples_), [=] (int col, double & innerUpdate) {
             innerUpdate += i_vec(col) * j_vec(col);
        }, e_ij);

        e_ij /= nsamples_;

        //e_kl
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team_member, nsamples_), [=] (int col, double & innerUpdate) {
             innerUpdate += k_vec(col) * l_vec(col);
        }, e_kl);

        e_kl /= nsamples_;

        //e_ik
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team_member, nsamples_), [=] (int col, double & innerUpdate) {
             innerUpdate += i_vec(col) * k_vec(col);
        }, e_ik);

        e_ik /= nsamples_;

        //e_jl
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team_member, nsamples_), [=] (int col, double & innerUpdate) {
             innerUpdate += j_vec(col) * l_vec(col);
        }, e_jl);

        e_jl /= nsamples_;

        //e_il
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team_member, nsamples_), [=] (int col, double & innerUpdate) {
             innerUpdate += i_vec(col) * l_vec(col);
        }, e_il);

        e_il /= nsamples_;

        //e_jk
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team_member, nsamples_), [=] (int col, double & innerUpdate) {
             innerUpdate += j_vec(col) * k_vec(col);
        }, e_jk);

        e_jk /= nsamples_;
        
        local_sub_matrix_(i,local_col) = (e_ijkl - e_ij*e_kl - e_ik*e_jl - e_il*e_jk) ;
 
      }


    
  }

};
