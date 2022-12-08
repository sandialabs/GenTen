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

#include "Genten_Util.hpp"

#include "Kokkos_UnorderedMap.hpp"

namespace Genten {

  namespace Impl {

    // Statically sized array for storing tensor indices as keys in UnorderedMap
    template <typename T, unsigned N>
    class Array {
    public:
      typedef T &                                 reference ;
      typedef typename std::add_const<T>::type &  const_reference ;
      typedef size_t                              size_type ;
      typedef ptrdiff_t                           difference_type ;
      typedef T                                   value_type ;
      typedef T *                                 pointer ;
      typedef typename std::add_const<T>::type *  const_pointer ;

      KOKKOS_INLINE_FUNCTION static constexpr size_type size() { return N ; }

      template< typename iType >
      KOKKOS_INLINE_FUNCTION
      reference operator[]( const iType & i ) { return x[i]; }

      template< typename iType >
      KOKKOS_INLINE_FUNCTION
      const_reference operator[]( const iType & i ) const { return x[i]; }

    private:
      T x[N];

    };

  }

  // A container for storing hash maps with various side Array<T,N> keys
  template <typename ExecSpace>
  class TensorHashMap {
  private:
    typedef Impl::Array<ttb_indx, 3> key_type_3;
    typedef Impl::Array<ttb_indx, 4> key_type_4;
    typedef Impl::Array<ttb_indx, 5> key_type_5;
    typedef Impl::Array<ttb_indx, 6> key_type_6;

    typedef Kokkos::UnorderedMap<key_type_3, ttb_real, ExecSpace> map_type_3;
    typedef Kokkos::UnorderedMap<key_type_4, ttb_real, ExecSpace> map_type_4;
    typedef Kokkos::UnorderedMap<key_type_5, ttb_real, ExecSpace> map_type_5;
    typedef Kokkos::UnorderedMap<key_type_6, ttb_real, ExecSpace> map_type_6;

  public:

    typedef typename map_type_3::size_type size_type;

    TensorHashMap() = default;

    TensorHashMap(ttb_indx ndim_, ttb_indx nnz) : ndim(ndim_) {
      if (ndim == 3) map_3 = map_type_3(nnz);
      else if (ndim == 4) map_4 = map_type_4(nnz);
      else if (ndim == 5) map_5 = map_type_5(nnz);
      else if (ndim == 6) map_6 = map_type_6(nnz);
      else Genten::error("Invalid tensor dimension for hash map!");
    }

    template <typename ind_t>
    KOKKOS_INLINE_FUNCTION
    void insert(const ind_t& ind, const ttb_real val) const {
      if (ndim == 3) insert_map(ind, val, map_3);
      else if (ndim == 4) insert_map(ind, val, map_4);
      else if (ndim == 5) insert_map(ind, val, map_5);
      else if (ndim == 6) insert_map(ind, val, map_6);
      return;
    }

    template <typename ind_t>
    KOKKOS_INLINE_FUNCTION
    bool exists(const ind_t& ind) const {
      if (ndim == 3) return exists_map(ind, map_3);
      else if (ndim == 4) return exists_map(ind, map_4);
      else if (ndim == 5) return exists_map(ind, map_5);
      else if (ndim == 6) return exists_map(ind, map_6);
      return false;
    }

    template <typename ind_t>
    KOKKOS_INLINE_FUNCTION
    size_type find(const ind_t& ind) const {
      if (ndim == 3) return find_map(ind, map_3);
      else if (ndim == 4) return find_map(ind, map_4);
      else if (ndim == 5) return find_map(ind, map_5);
      else if (ndim == 6) return find_map(ind, map_6);
      return 0;
    }

    KOKKOS_INLINE_FUNCTION
    bool valid_at(size_type i) const {
      if (ndim == 3) return map_3.valid_at(i);
      else if (ndim == 4) return map_4.valid_at(i);
      else if (ndim == 5) return map_5.valid_at(i);
      else if (ndim == 6) return map_6.valid_at(i);
      return false;
    }

    KOKKOS_INLINE_FUNCTION
    ttb_real value_at(size_type i) const {
      if (ndim == 3) return map_3.value_at(i);
      else if (ndim == 4) return map_4.value_at(i);
      else if (ndim == 5) return map_5.value_at(i);
      else if (ndim == 6) return map_6.value_at(i);
      return 0;
    }

    void print_histogram(std::ostream& out) {
      if (ndim == 3) print_histogram_map(out, map_3);
      else if (ndim == 4) print_histogram_map(out, map_4);
      else if (ndim == 5) print_histogram_map(out, map_5);
      else if (ndim == 6) print_histogram_map(out, map_6);
      return;
    }

  private:

    ttb_indx ndim;
    map_type_3 map_3;
    map_type_4 map_4;
    map_type_5 map_5;
    map_type_6 map_6;

    template <typename ind_t, typename map_type>
    KOKKOS_INLINE_FUNCTION
    void insert_map(const ind_t& ind, const ttb_real val, map_type& map) const {
      typename map_type::key_type key;
      for (ttb_indx i=0; i<ndim; ++i)
        key[i] = ind[i];
      if (map.insert(key,val).failed())
        Kokkos::abort("Hash map insert failed!");
    }

    template <typename ind_t, typename map_type>
    KOKKOS_INLINE_FUNCTION
    bool exists_map(const ind_t& ind, map_type& map) const {
      typename map_type::key_type key;
      for (ttb_indx i=0; i<ndim; ++i)
        key[i] = ind[i];
      return map.exists(key);
    }

    template <typename ind_t, typename map_type>
    KOKKOS_INLINE_FUNCTION
    size_type find_map(const ind_t& ind, map_type& map) const {
      typename map_type::key_type key;
      for (ttb_indx i=0; i<ndim; ++i)
        key[i] = ind[i];
      return map.find(key);
    }

    template <typename map_type>
    void print_histogram_map(std::ostream& out, map_type& map) const {
      /* Currently does not compile on Cuda with UVM enabled, so commenting
       * out for now.
      auto h = map.get_histogram();
      h.calculate();
      out << "length:" << std::endl;
      h.print_length(out);
      out << "distance:" << std::endl;
      h.print_distance(out);
      out << "block distance:" << std::endl;
      h.print_block_distance(out);
      */
      Genten::error("histogram is currently disabled!");
    }
  };

}
