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
#include<Kokkos_Core.hpp>
//HK#include<KokkosBlas3_gemm.hpp>
//HK#include <KokkosBatched_Gemm_Decl.hpp>
//HK#include <KokkosBatched_Gemm_Team_Impl.hpp>

//HK#using namespace KokkosBatched;

template <typename ExecSpace>
struct ComputeKhatriRaoProduct{
  typedef typename Kokkos::View<double*, ExecSpace> sub_view_type;
  typedef typename sub_view_type::device_type device_type;
  typedef typename Kokkos::TeamPolicy<ExecSpace>::member_type member_type;
  typedef typename Kokkos::View<double**, Kokkos::LayoutLeft, ExecSpace> view_type;

  int nrows_;
  int ncols_;
  int nteams_x_;
  int nteams_y_;
  view_type input_raw_data_;
  view_type krp_result_;

  ComputeKhatriRaoProduct(int nrows, int ncols, int nteams_x, int nteams_y, const view_type& inp_raw_data, view_type& krp_result):
    nrows_(nrows), ncols_(ncols), nteams_x_(nteams_x), nteams_y_(nteams_y),
    input_raw_data_(inp_raw_data), krp_result_(krp_result)
  {}
  KOKKOS_INLINE_FUNCTION void
  operator() (const member_type team_member) const{
    //HKtypedef Mode::Team mode_type;
    //HKtypedef Algo::Level3::Unblocked algo_type;

    //Figure out the team_idx and team_idy indices based on flattened team rank 
    int team_rank = team_member.league_rank();
    int team_idy = team_rank / nteams_x_;
    int team_idx = team_rank % nteams_x_; 

    //HK if(team_member.team_rank()==0){printf("Team %d j/k/l are: %d/%d/%d\n",team_member.league_rank(),j,k,l_);}

    int nthreads_per_team = team_member.team_size();

    //Each team is going to handle a subset of columns, with some handling one extra column, if they are not exactly divisible by nteams_y
    int cols_per_block = ncols_ / nteams_y_;
    int blocks_with_extra_col = ncols_ % nteams_y_;

    int col_start;

    if(team_idy < blocks_with_extra_col) {//Lucky teams that process an extra column
      cols_per_block += 1;
      col_start = cols_per_block*team_idy;
    } else {
      int cols_assigned_to_lower_blocks = (cols_per_block + 1) * blocks_with_extra_col;
      col_start = cols_assigned_to_lower_blocks + (team_idy - blocks_with_extra_col) * cols_per_block;
    }

    int col_end = col_start + cols_per_block - 1;

    
    int row = team_member.team_rank() + team_idx * nthreads_per_team;

    int left_vec_row_idx = row/nrows_; 
    int right_vec_row_idx = row % nrows_;

    for(int col = col_start; col <= col_end; col++) {
       //int col_offset = col * nrows_ * nrows_;
       if (row < nrows_ * nrows_)
           krp_result_(row, col) = input_raw_data_(left_vec_row_idx, col) * input_raw_data_(right_vec_row_idx, col);
           //B[col_offset + row] = A[col*R + left_vec_row_idx] * A[col*R + right_vec_row_idx];
    }

    
  }

};
