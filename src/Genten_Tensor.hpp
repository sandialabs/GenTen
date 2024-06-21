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

#include <cassert>
#include <any>

#include "Genten_Array.hpp"
#include "Genten_IndxArray.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_DistTensor.hpp"

namespace Genten {

namespace Impl {

// Class for tensor multi-dimensional indexing using "layout left" ordering,
// i.e., "Fortran" ordering
struct TensorLayoutLeft {
public:

  // Convert subscript to linear index
  template <typename SubType, typename ExecSpace>
  KOKKOS_INLINE_FUNCTION
  static ttb_indx sub2ind(const SubType& sub,
                          const IndxArrayT<ExecSpace>& siz)
  {
    const ttb_indx nd = siz.size();
    for (ttb_indx i=0; i<nd; ++i)
      assert(sub[i] < siz[i]);

    ttb_indx idx = 0;
    ttb_indx cumprod = 1;
    for (ttb_indx i=0; i<nd; ++i) {
      idx += sub[i] * cumprod;
      cumprod *= siz[i];
    }
    return idx;
  }

  // Convert global_subscript to linear index
  template <typename SubType, typename ExecSpace>
  KOKKOS_INLINE_FUNCTION
  static ttb_indx global_sub2ind(const SubType& sub,
                                 const IndxArrayT<ExecSpace>& siz,
                                 const IndxArrayT<ExecSpace>& lower)
  {
    const ttb_indx nd = siz.size();
    assert(lower.size() == nd);
    for (ttb_indx i=0; i<nd; ++i) {
      assert(sub[i] >= lower[i]);
      assert(sub[i]-lower[i] < siz[i]);
    }

    ttb_indx idx = 0;
    ttb_indx cumprod = 1;
    for (ttb_indx i=0; i<nd; ++i) {
      idx += (sub[i]-lower[i]) * cumprod;
      cumprod *= siz[i];
    }
    return idx;
  }

  // Convert linear index to subscript
  template <typename SubType, typename ExecSpace>
  KOKKOS_INLINE_FUNCTION
  static void ind2sub(SubType& sub, const IndxArrayT<ExecSpace>& siz,
                      ttb_indx cumprod, ttb_indx ind)
  {
    const ttb_indx nd = siz.size();
    assert(ind < cumprod);

    ttb_indx sbs;
    for (ttb_indx i=nd; i>0; --i) {
      cumprod = cumprod / siz[i-1];
      sbs = ind / cumprod;
      sub[i-1] = sbs;
      ind = ind - (sbs * cumprod);
    }
  }

  template <typename SubType, typename ExecSpace>
  KOKKOS_INLINE_FUNCTION
  static bool increment_sub(SubType& sub, const IndxArrayT<ExecSpace>& siz,
                            const ttb_indx n)
  {
    const ttb_indx nd = siz.size();
    const ttb_indx first_mode = n == 0 ? 1 : 0;
    const ttb_indx last_mode = n == nd-1 ? nd-1 : nd;
    ++sub[first_mode];
    for (ttb_indx i=first_mode; i<last_mode; ++i) {
      if (i == n)
        continue;
      if (sub[i] != siz[i])
        break;
      else if (i < (last_mode-1)) {
        sub[i] = 0;
        if (i+1 != n)
          ++sub[i+1];
        else if (i < (last_mode-2))
          ++sub[i+2];
      }
    }
    if (sub[last_mode-1] == siz[last_mode-1])
      return false;
    return true;
  }
};

// Class for tensor multi-dimensional indexing using "layout right" ordering,
// i.e., "C" ordering
struct TensorLayoutRight {
public:

  // Convert subscript to linear index
  template <typename SubType, typename ExecSpace>
  KOKKOS_INLINE_FUNCTION
  static ttb_indx sub2ind(const SubType& sub,
                          const IndxArrayT<ExecSpace>& siz)
  {
    const ttb_indx nd = siz.size();
    for (ttb_indx i=0; i<nd; ++i)
      assert(sub[i] < siz[i]);

    ttb_indx idx = 0;
    ttb_indx cumprod = 1;
    for (ttb_indx i=nd; i>0; --i) {
      idx += sub[i-1] * cumprod;
      cumprod *= siz[i-1];
    }
    return idx;
  }

  // Convert global_subscript to linear index
  template <typename SubType, typename ExecSpace>
  KOKKOS_INLINE_FUNCTION
  static ttb_indx global_sub2ind(const SubType& sub,
                                 const IndxArrayT<ExecSpace>& siz,
                                 const IndxArrayT<ExecSpace>& lower)
  {
    const ttb_indx nd = siz.size();
    assert(lower.size() == nd);
    for (ttb_indx i=0; i<nd; ++i) {
      assert(sub[i] >= lower[i]);
      assert(sub[i]-lower[i] < siz[i]);
    }

    ttb_indx idx = 0;
    ttb_indx cumprod = 1;
    for (ttb_indx i=nd; i>0; --i) {
      idx += (sub[i-1]-lower[i-1]) * cumprod;
      cumprod *= siz[i-1];
    }
    return idx;
  }

  // Convert linear index to subscript
  template <typename SubType, typename ExecSpace>
  KOKKOS_INLINE_FUNCTION
  static void ind2sub(SubType& sub, const IndxArrayT<ExecSpace>& siz,
                      ttb_indx cumprod, ttb_indx ind)
  {
    const ttb_indx nd = siz.size();
    assert(ind < cumprod);

    ttb_indx sbs;
    for (ttb_indx i=0; i<nd; ++i) {
      cumprod = cumprod / siz[i];
      sbs = ind / cumprod;
      sub[i] = sbs;
      ind = ind - (sbs * cumprod);
    }
  }

  template <typename SubType, typename ExecSpace>
  KOKKOS_INLINE_FUNCTION
  static bool increment_sub(SubType& sub, const IndxArrayT<ExecSpace>& siz,
                            const ttb_indx n)
  {
    const ttb_indx nd = siz.size();
    const ttb_indx first_mode = n == nd-1 ? nd-2 : nd-1;
    const ttb_indx last_mode = n == 0 ? 1 : 0;
    ++sub[first_mode];
    for (ttb_indx i=first_mode+1; i>last_mode; --i) {
      if (i-1 == n)
        continue;
      if (sub[i-1] != siz[i-1])
        break;
      else if (i-1 > last_mode) {
        sub[i-1] = 0;
        if (i-2 != n)
          ++sub[i-2];
        else if (i-1 > (last_mode+1))
          ++sub[i-3];
      }
    }
    if (sub[last_mode] == siz[last_mode])
      return false;
    return true;
  }
};

}

template <typename ExecSpace> class SptensorT;

/* The Genten::Tensor class stores dense tensors. These are stored as
 * flat arrays using the Genten::Array class.
 */

template <typename ExecSpace> class TensorT;
typedef TensorT<DefaultHostExecutionSpace> Tensor;

template <typename ExecSpace, typename Layout>
class TensorImpl
{

public:

  typedef ExecSpace exec_space;
  typedef typename ArrayT<ExecSpace>::host_mirror_space host_mirror_space;
  typedef TensorImpl<host_mirror_space,Layout> HostMirror;
  typedef Layout layout_type;

  // Empty construtor.
  KOKKOS_DEFAULTED_FUNCTION
  TensorImpl() = default;

  // Copy constructor
  KOKKOS_DEFAULTED_FUNCTION
  TensorImpl(const TensorImpl& src) = default;

  // Construct tensor of given size initialized to val.
  TensorImpl(const IndxArrayT<ExecSpace>& sz, ttb_real val = 0.0) :
    siz(sz.clone()),
    lower_bound(sz.size(),ttb_indx(0)),
    upper_bound(siz.clone()) {
    siz_host = create_mirror_view(siz);
    deep_copy(siz_host, siz);
    values = ArrayT<ExecSpace>(siz_host.prod(), val);
  }

  // Construct tensor with given size and values
  TensorImpl(const IndxArrayT<ExecSpace>& sz,
             const ArrayT<ExecSpace>& vals) :
    siz(sz), values(vals),
    lower_bound(sz.size(),ttb_indx(0)),
    upper_bound(siz.clone()) {
    siz_host = create_mirror_view(siz);
    deep_copy(siz_host, siz);
  }

  // Construct tensor with given size and values
  TensorImpl(const IndxArrayT<ExecSpace>& sz,
             const ArrayT<ExecSpace>& vals,
             const IndxArrayT<ExecSpace>& l,
             const IndxArrayT<ExecSpace>& u) :
    siz(sz), values(vals),
    lower_bound(l),
    upper_bound(u) {
    siz_host = create_mirror_view(siz);
    deep_copy(siz_host, siz);
  }

  // Construct tensor for Sptensor
  TensorImpl(const SptensorT<ExecSpace>& src);

  // Construct tensor for Ktensor
  TensorImpl(const KtensorT<ExecSpace>& src);

  // Destructor.
  KOKKOS_DEFAULTED_FUNCTION
  ~TensorImpl() = default;

  // Copy another tensor (shallow copy)
  TensorImpl& operator=(const TensorImpl& src) = default;

  // Return the number of dimensions (i.e., the order).
  KOKKOS_INLINE_FUNCTION
  ttb_indx ndims() const { return siz.size(); }

  // Return size of dimension i.
  KOKKOS_INLINE_FUNCTION
  ttb_indx size(ttb_indx i) const {
    KOKKOS_IF_ON_DEVICE(return siz[i];)
    KOKKOS_IF_ON_HOST(return siz_host[i];)
  }

  // Return the entire size array.
  KOKKOS_INLINE_FUNCTION
  const IndxArrayT<ExecSpace>& size() const { return siz; }

  // Return the entire size array.
  const IndxArrayT<host_mirror_space>& size_host() const { return siz_host; }

  // Return the total number of elements in the tensor.
  KOKKOS_INLINE_FUNCTION
  ttb_indx numel() const { return values.size(); }

   KOKKOS_INLINE_FUNCTION
   ttb_real numel_float() const { return ttb_real(numel()); }

  // Convert subscript to linear index
  template <typename SubType>
  KOKKOS_INLINE_FUNCTION
  ttb_indx sub2ind(const SubType& sub) const {
    return layout_type::sub2ind(sub, siz);
  }

  // Convert global subscript to linear index
  template <typename SubType>
  KOKKOS_INLINE_FUNCTION
  ttb_indx global_sub2ind(const SubType& sub) const {
    return layout_type::global_sub2ind(sub, siz, lower_bound);
  }

   // Convert linear index to subscript
  template <typename SubType>
  KOKKOS_INLINE_FUNCTION
  void ind2sub(SubType& sub, ttb_indx ind) const {
    ttb_indx cumprod = values.size();
    return layout_type::ind2sub(sub, siz, cumprod, ind);
  }

  // Return the i-th linearly indexed element.
  KOKKOS_INLINE_FUNCTION
  ttb_real & operator[](ttb_indx i) const { return values[i]; }

  // Return the element indexed by the given subscript array.
  KOKKOS_INLINE_FUNCTION
  ttb_real& operator[](const IndxArrayT<ExecSpace>& sub) const {
    return values[sub2ind(sub)];
  }

  // Allow indexing of the form tensor(i1,...,in)
  template <typename...Args>
  ttb_real& operator()(Args...args) const {
    assert(sizeof...(args) == siz.size());
    IndxArrayT<ExecSpace> sub = {args...};
    return (*this)[sub];
  }

  // Increment subscript to the next subscript lexicographically holding
  // the given mode fixed. Returns false if there are no more subscripts
  template <typename SubType>
  KOKKOS_INLINE_FUNCTION
  bool increment_sub(SubType& sub, const ttb_indx n) const {
    return layout_type::increment_sub(sub, siz, n);
  }

  // Return the number of elements with a nonzero value.
  // We don't actually look at the values, and assume all are nonzero.
  KOKKOS_INLINE_FUNCTION
  ttb_indx nnz() const { return values.size(); }

  // Return the norm (sqrt of the sum of the squares of all entries).
  ttb_real norm() const { return values.norm(NormTwo); }

  // Return const reference to values array
  KOKKOS_INLINE_FUNCTION
  const ArrayT<ExecSpace>& getValues() const { return values; }

  KOKKOS_INLINE_FUNCTION
  ttb_indx lowerBound(const unsigned n) const {
    assert(n < lower_bound.size());
    return lower_bound[n];
  }
  KOKKOS_INLINE_FUNCTION
  ttb_indx upperBound(const unsigned n) const {
    assert(n < upper_bound.size());
    return upper_bound[n];
  }
  KOKKOS_INLINE_FUNCTION
  IndxArrayT<ExecSpace> getLowerBounds() const { return lower_bound; }
  KOKKOS_INLINE_FUNCTION
  IndxArrayT<ExecSpace> getUpperBounds() const { return upper_bound; }

protected:

  // Size of the tensor
  IndxArrayT<ExecSpace> siz;
  IndxArrayT<host_mirror_space> siz_host;

  // Entries of the tensor
  // TBD describe storage order via ind2sub and sub2ind
  ArrayT<ExecSpace> values;

  // Lower and upper bounds of tensor global indices
  IndxArrayT<ExecSpace> lower_bound;
  IndxArrayT<ExecSpace> upper_bound;
};

enum class TensorLayout {
  Left,
  Right
};

template <typename ExecSpace>
class TensorT : public DistTensor<ExecSpace>
{
public:

  using left_impl_type = TensorImpl<ExecSpace,Impl::TensorLayoutLeft>;
  using right_impl_type = TensorImpl<ExecSpace,Impl::TensorLayoutRight>;
  using dist_type = DistTensor<ExecSpace>;
  using exec_space = typename left_impl_type::exec_space;
  using host_mirror_space = typename left_impl_type::host_mirror_space;
  using HostMirror = TensorT<host_mirror_space>;

  TensorT(TensorLayout layout = TensorLayout::Left) :
    dist_type(), layout_(layout) {}

  TensorT(const IndxArrayT<ExecSpace>& sz, ttb_real val = 0.0,
          TensorLayout layout = TensorLayout::Left) :
    dist_type(sz.size()), layout_(layout)
  {
    if (layout == TensorLayout::Left)
      left_impl_ = left_impl_type(sz,val);
    else
      right_impl_ = right_impl_type(sz,val);
  }
  TensorT(const IndxArrayT<ExecSpace>& sz, const ArrayT<ExecSpace>& vals,
          TensorLayout layout = TensorLayout::Left) :
    dist_type(sz.size()), layout_(layout)
  {
    if (layout == TensorLayout::Left)
      left_impl_ = left_impl_type(sz,vals);
    else
      right_impl_ = right_impl_type(sz,vals);
  }
  TensorT(const IndxArrayT<ExecSpace>& sz, const ArrayT<ExecSpace>& vals,
          const IndxArrayT<ExecSpace>& l, const IndxArrayT<ExecSpace>& u,
          TensorLayout layout = TensorLayout::Left) :
    dist_type(sz.size()), layout_(layout)
  {
    if (layout == TensorLayout::Left)
      left_impl_ = left_impl_type(sz,vals,l,u);
    else
      right_impl_ = right_impl_type(sz,vals,l,u);
  }
  TensorT(const SptensorT<ExecSpace>& src,
          TensorLayout layout = TensorLayout::Left) :
    dist_type(src.ndims()), layout_(layout)
  {
    if (layout == TensorLayout::Left)
      left_impl_ = left_impl_type(src);
    else
      right_impl_ = right_impl_type(src);
  }
  TensorT(const KtensorT<ExecSpace>& src,
          TensorLayout layout = TensorLayout::Left) :
    dist_type(src.ndims()), layout_(layout)
  {
    if (layout == TensorLayout::Left)
      left_impl_ = left_impl_type(src);
    else
      right_impl_ = right_impl_type(src);
  }
  TensorT(const left_impl_type& impl) :
    dist_type(impl.ndims()), layout_(TensorLayout::Left), left_impl_(impl), right_impl_() {}
  TensorT(const right_impl_type& impl) :
    dist_type(impl.ndims()), layout_(TensorLayout::Right), left_impl_(), right_impl_(impl) {}

  TensorT(TensorT&&) = default;
  TensorT(const TensorT&) = default;
  TensorT& operator=(TensorT&&) = default;
  TensorT& operator=(const TensorT&) = default;
  ~TensorT() = default;

  TensorLayout getLayout() const { return layout_; }
  bool has_left_impl() const { return layout_ == TensorLayout::Left; }
  bool has_right_impl() const { return layout_ == TensorLayout::Right; }
  left_impl_type& left_impl() { return left_impl_; }
  const left_impl_type& left_impl() const { return left_impl_; }
  right_impl_type& right_impl() { return right_impl_; }
  const right_impl_type& right_impl() const { return right_impl_; }

  // Create a new tensor containing the transpose with the supplied layout
  TensorT transpose(TensorLayout new_layout) const;

  // Create a new tensor switching to the new layout
  TensorT switch_layout(TensorLayout new_layout) const;

  ttb_indx ndims() const {
    return layout_ == TensorLayout::Left ? left_impl_.ndims() : right_impl_.ndims();
  }

  ttb_indx size(ttb_indx i) const {
    return layout_ == TensorLayout::Left ? left_impl_.size(i) : right_impl_.size(i);
  }
  const IndxArrayT<ExecSpace>& size() const {
    return layout_ == TensorLayout::Left ? left_impl_.size() : right_impl_.size();
  }
  const IndxArrayT<host_mirror_space>& size_host() const {
    return layout_ == TensorLayout::Left ? left_impl_.size_host() : right_impl_.size_host();
  }

  ttb_indx numel() const {
    return layout_ == TensorLayout::Left ? left_impl_.numel() : right_impl_.numel();
  }
  ttb_real numel_float() const {
    return layout_ == TensorLayout::Left ? left_impl_.numel_float() : right_impl_.numel_float();
  }

  template <typename SubType>
  ttb_indx sub2ind(const SubType& sub) const {
    return layout_ == TensorLayout::Left ? left_impl_.sub2ind(sub) : right_impl_.sub2ind(sub);
  }
  template <typename SubType>
  ttb_indx global_sub2ind(const SubType& sub) const {
    return layout_ == TensorLayout::Left ? left_impl_.global_sub2ind(sub) : right_impl_.global_sub2ind(sub);
  }
  template <typename SubType>
  void ind2sub(SubType& sub, ttb_indx ind) const {
    layout_ == TensorLayout::Left ? left_impl_.ind2sub(sub,ind) : right_impl_.ind2sub(sub,ind);
  }

  ttb_real & operator[](ttb_indx i) const {
    return layout_ == TensorLayout::Left ? left_impl_[i] : right_impl_[i];
  }
  ttb_real& operator[](const IndxArrayT<ExecSpace>& sub) const {
    return layout_ == TensorLayout::Left ? left_impl_[sub] : right_impl_[sub];
  }
  template <typename...Args>
  ttb_real& operator()(Args...args) const {
    return layout_ == TensorLayout::Left ? left_impl_(args...) : right_impl_(args...);
  }

  template <typename SubType>
  bool increment_sub(SubType& sub, const ttb_indx n) const {
    return layout_ == TensorLayout::Left ? left_impl_.increment_sub(sub,n) : right_impl_.increment_sub(sub,n);
  }

  ttb_indx nnz() const {
    return layout_ == TensorLayout::Left ? left_impl_.nnz() : right_impl_.nnz();
  }
  ttb_real norm() const {
    return layout_ == TensorLayout::Left ? left_impl_.norm() : right_impl_.norm();
  }
  const ArrayT<ExecSpace>& getValues() const {
    return layout_ == TensorLayout::Left ? left_impl_.getValues() : right_impl_.getValues();
  }

  ttb_indx lowerBound(const unsigned n) const {
    return layout_ == TensorLayout::Left ? left_impl_.lowerBound(n) : right_impl_.lowerBound(n);
  }
  ttb_indx upperBound(const unsigned n) const {
    return layout_ == TensorLayout::Left ? left_impl_.upperBound(n) : right_impl_.upperBound(n);
  }
  IndxArrayT<ExecSpace> getLowerBounds() const {
    return layout_ == TensorLayout::Left ? left_impl_.getLowerBounds() : right_impl_.getLowerBounds();
  }
  IndxArrayT<ExecSpace> getUpperBounds() const {
    return layout_ == TensorLayout::Left ? left_impl_.getUpperBounds() : right_impl_.getUpperBounds();
  }

  ttb_indx global_numel() const
  {
    ttb_indx my_numel = numel();
    const ProcessorMap *pmap = this->getProcessorMap();
    if (pmap != nullptr)
      my_numel = pmap->gridAllReduce(my_numel);
    return my_numel;
  }
  ttb_real global_numel_float() const
  {
    ttb_real my_numel = numel_float();
    const ProcessorMap *pmap = this->getProcessorMap();
    if (pmap != nullptr)
      my_numel = pmap->gridAllReduce(my_numel);
    return my_numel;
  }
  ttb_indx global_nnz() const
  {
    ttb_indx my_nnz = nnz();
    const ProcessorMap *pmap = this->getProcessorMap();
    if (pmap != nullptr)
      my_nnz = pmap->gridAllReduce(my_nnz);
    return my_nnz;
  }
  ttb_real global_norm() const
  {
    ttb_real nrm = norm();
    nrm = nrm*nrm;
    const ProcessorMap *pmap = this->getProcessorMap();
    if (pmap != nullptr)
      nrm = this->pmap->gridAllReduce(nrm);
    return std::sqrt(nrm);
  }

  // For passing extra data, like numpy arrays, through
  template <typename T>
  void set_extra_data(const T& a) {
    extra_data = std::make_any<T>(a);
  }
  bool has_extra_data() const {
    return extra_data.has_value();
  }
  template <typename T>
  bool has_extra_data_type() const {
    return extra_data.has_value() && (std::any_cast<T>(&extra_data) != nullptr);
  }
  template <typename T>
  T get_extra_data() const {
    gt_assert(extra_data.has_value());
    return std::any_cast<T>(extra_data);
  }
  template <typename E>
  void copy_extra_data(const TensorT<E>& x) {
    // only copy extra data if this and x point to the same data
    if (this->getValues().ptr() == x.getValues().ptr())
      extra_data = x.extra_data;
  }

  // DistTensor methods
  virtual bool isSparse() const override { return false; }
  virtual bool isDense() const override { return true; }
  virtual SptensorT<ExecSpace> getSptensor() const override {
    return SptensorT<ExecSpace>();
  }
  virtual TensorT<ExecSpace> getTensor() const override { return *this; }

protected:

  TensorLayout layout_;
  left_impl_type left_impl_;
  right_impl_type right_impl_;

  std::any extra_data;
  template <typename E> friend class TensorT;

};

template <typename ExecSpace>
typename TensorT<ExecSpace>::HostMirror
create_mirror_view(const TensorT<ExecSpace>& a)
{
  typedef typename TensorT<ExecSpace>::HostMirror HostMirror;
  HostMirror hm( create_mirror_view(a.size()),
                 create_mirror_view(a.getValues()),
                 a.getLayout() );
  hm.copy_extra_data(a);
  return hm;
}

template <typename Space, typename ExecSpace>
TensorT<Space>
create_mirror_view(const Space& s, const TensorT<ExecSpace>& a)
{
  TensorT<Space> t( create_mirror_view(s, a.size()),
                    create_mirror_view(s, a.getValues()),
                    create_mirror_view(s, a.getLowerBounds()),
                    create_mirror_view(s, a.getUpperBounds()),
                    a.getLayout() );
  t.copy_extra_data(a);
  return t;
}

template <typename E1, typename E2>
void deep_copy(TensorT<E1>& dst, const TensorT<E2>& src)
{
  deep_copy( dst.size(), src.size() );
  deep_copy( dst.size_host(), src.size_host() );
  deep_copy( dst.getValues(), src.getValues() );
  deep_copy( dst.getLowerBounds(), src.getLowerBounds() );
  deep_copy( dst.getUpperBounds(), src.getUpperBounds() );
}

}
