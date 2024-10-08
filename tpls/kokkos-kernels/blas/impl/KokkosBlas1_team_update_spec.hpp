//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOSBLAS1_TEAM_UPDATE_SPEC_HPP_
#define KOKKOSBLAS1_TEAM_UPDATE_SPEC_HPP_

#include <KokkosKernels_config.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_ArithTraits.hpp>
#include <Kokkos_InnerProductSpaceTraits.hpp>

namespace KokkosBlas {
namespace Experimental {
namespace Impl {

template <class XV, class YV, class ZV>
struct team_update_tpl_spec_avail {
  constexpr static bool value = false;
};

// Unification and Specialization layer
template <class TeamType, class XVector, class YVector, class ZVector,
          bool tpl_spec_avail = team_update_tpl_spec_avail<XVector, YVector, ZVector>::value>
struct TeamUpdate {
  static KOKKOS_INLINE_FUNCTION void team_update(const TeamType& team,
                                                 const typename XVector::non_const_value_type& alpha, const XVector& x,
                                                 const typename YVector::non_const_value_type& beta, const YVector& y,
                                                 const typename ZVector::non_const_value_type& gamma, const ZVector& z);
};

template <class TeamType, class XVector, class YVector, class ZVector>
struct TeamUpdate<TeamType, XVector, YVector, ZVector, false> {
  static KOKKOS_INLINE_FUNCTION void team_update(const TeamType& team,
                                                 const typename XVector::non_const_value_type& alpha, const XVector& x,
                                                 const typename YVector::non_const_value_type& beta, const YVector& y,
                                                 const typename ZVector::non_const_value_type& gamma,
                                                 const ZVector& z) {
    const int N = x.extent(0);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N),
                         [&](const int& i) { z(i) = gamma * z(i) + alpha * x(i) + beta * y(i); });
  }
};

}  // namespace Impl
}  // namespace Experimental
}  // namespace KokkosBlas

#endif
