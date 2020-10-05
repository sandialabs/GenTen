#@HEADER
# ************************************************************************
#     Genten: Software for Generalized Tensor Decompositions
#     by Sandia National Laboratories
#
# Sandia National Laboratories is a multimission laboratory managed
# and operated by National Technology and Engineering Solutions of Sandia,
# LLC, a wholly owned subsidiary of Honeywell International, Inc., for the
# U.S. Department of Energy's National Nuclear Security Administration under
# contract DE-NA0003525.
#
# Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ************************************************************************
#@HEADER

# Disallow compiler versions that are known to not work (we can't rely on
# Kokkos' checks because some compilers that work with Kokkos do not work with
# GenTen).
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.0")
    message(FATAL_ERROR "GenTen: insufficient GCC compiler version ${CMAKE_CXX_COMPILER_VERSION} (requires GCC 6.0 or later)")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "19.0")
    message(FATAL_ERROR "GenTen: insufficient Intel compiler version ${CMAKE_CXX_COMPILER_VERSION} (requires Intel 19 or later)")
  endif()
endif()