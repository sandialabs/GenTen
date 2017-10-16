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

/*!
  @file Genten_SpRowSubprob.h
  @brief Encapsulate a sparse row subproblem for CP-APR algorithms.
*/

#pragma once

#include "Genten_Array.hpp"
#include "Genten_IndxArray.hpp"
#include "Genten_RandomMT.hpp"
#include "Genten_Util.hpp"
#include <assert.h>

namespace Genten
{

/*! @class Genten::FacMatrix
 *  @brief  Dense matrix usually serving as a Ktensor factor matrix.
 *
 *  This class stores a dense matrix, with operations specialized to the
 *  factor matrices of a Ktensor.  Rows correspond to the indices of a
 *  data tensor mode, and columns correspond to the number of components
 *  in the Ktensor.
 *
 *  Matrix elements are stored in row major order, meaning the (i,j) entry
 *  is stored at [i * ncols + j].  Grey Ballard did some analysis to justify
 *  row major instead of column major.  It optimizes computations in
 *  rowTimes(), which is called during the innermost loop of an mttkrp
 *  computation.  The ordering is mostly hidden from other classes.
 */

class FacMatArray;   // Forward declaration to avoid circular referencing
class Tensor;        // Forward declaration for friend declaration
class Sptensor;      // Forward declaration for friend declaration

class FacMatrix
{
public:

    /** ----------------------------------------------------------------
     *  @name Constructors and Destructors
     *  @{
     *  ---------------------------------------------------------------- */

    //! Default constructor.
    KOKKOS_INLINE_FUNCTION
    FacMatrix() = default;

    //! Constructor to create an uninitialized matrix of size M x N.
    /*!
     *  @param[in] m  Number of rows, should equal the size of a given mode
     *                in the Ktensor.
     *  @param[in] n  Number of columns, should equal the number of components
     *                in the Ktensor.
     */
    FacMatrix(ttb_indx m, ttb_indx n) :
      //data("Genten::FacMatrix::data",m,n)
      data(Kokkos::view_alloc("Genten::FacMatrix::data", Kokkos::AllowPadding),m,n)
 {}

    //! Constructor to create a Factor Matrix of Size M x N using the
    // given view
    template <typename T, typename ... P>
    FacMatrix(ttb_indx m, ttb_indx n, const Kokkos::View<T,P...>& v) :
      data(v) {}

    // Constructor to create a Factor Matrix of Size M x N using the
    // given data vector CVEC which is assumed to be stored *columnwise*.
    // Not currently used; therefore not part of doxygen API.
    FacMatrix(ttb_indx m, ttb_indx n, const ttb_real * cvec);

    //! Copy Constructor.
    KOKKOS_INLINE_FUNCTION
    FacMatrix(const FacMatrix & src) = default;

    //! Destructor.
    KOKKOS_INLINE_FUNCTION
    ~FacMatrix() = default;

    /** @} */

    /** ----------------------------------------------------------------
     *  @name Methods to Resize and Set All Elements.
     *  @{
     *  ---------------------------------------------------------------- */

    // Make a copy of an existing array.
    KOKKOS_INLINE_FUNCTION
    FacMatrix & operator=(const FacMatrix & src) = default;

    //! Set all entries to the given value.
    void operator= (ttb_real val);

    //! Deep copy between factor matrices
    void deep_copy(const FacMatrix& src) {
      assert(data.dimension_0() == src.data.dimension_0());
      assert(data.dimension_1() == src.data.dimension_1());
      Kokkos::deep_copy(data, src.data);
    }

    //! Set all entries to random values drawn uniformly from [0,1).
    /*!
     *  A new stream of Mersenne twister random numbers is generated, starting
     *  from an arbitrary seed value.  Use scatter() for reproducibility.
     */
    void rand();

    //! Set all entries to reproducible random values drawn uniformly from [0,1).
    /*!
     *  @param[in] bUseMatlabRNG  If true, then generate random samples
     *                            consistent with Matlab (costs twice as much
     *                            compared with no Matlab consistency).
     *  @param[in] cRMT           Mersenne Twister random number generator.
     *                            The seed should already be set.
     */
    void scatter (const bool        bUseMatlabRNG,
                        RandomMT &  cRMT);

    // Copy data from the column-oriented data array into a matrix of size m x n. 
    // Assumes that cvec is an array of length m*n.
    // Not currently used; therefore not part of doxygen API.
    void convertFromCol(ttb_indx m, ttb_indx n, const ttb_real * cvec);

    /** @} */

    /** ----------------------------------------------------------------
     *  @name Get and Set Methods
     *  @{
     *  ---------------------------------------------------------------- */

    //! Return the number of rows.
    KOKKOS_INLINE_FUNCTION
    ttb_indx nRows() const
    {
      return data.dimension_0();
    }

    //! Return the number of columns.
    KOKKOS_INLINE_FUNCTION
    ttb_indx nCols() const
    {
      return data.dimension_1();
    }

    //! Return a read-only entry (i,j).
    /*!
     *  @param[in] i  Row index.
     *  @param[in] j  Column index.
     */
    template <typename IType, typename JType>
    KOKKOS_INLINE_FUNCTION
    ttb_real & entry(IType i, JType j) const
    {
      assert((i < data.dimension_0()) && (j < data.dimension_1()));
      return data(i,j);
    }

    //! Number of of ttb_real elements stored in this object.
    KOKKOS_INLINE_FUNCTION
    ttb_indx reals() const
    {
      return data.size();
    }

    /** @} */

    /** ----------------------------------------------------------------
     *  @name Methods for Computations
     *  @{
     *  ---------------------------------------------------------------- */


    // ----- FUNCTIONS -----

    // Return true if this matrix is equal to b within the specified tolerance
    /* Being equal means that the two matrices are the same size and

            fabs(a(i,j) - b(i,j))
       ---------------------------------   < TOL
       max(1, fabs(a(i,j)), fabs(b(i,j))

       for all i,j.
    */
    bool isEqual(const Genten::FacMatrix & b, ttb_real tol) const;

    // Compute X = a * X
    void times(ttb_real a);

    // x += y
    /* accumulate y into x */
    void plus(const Genten::FacMatrix & y);

    // x += yi forall yi in ya
    void plusAll(const Genten::FacMatArray & ya);

    // X = Y'.
    /* Set this matrix equal to the transpose of the input matrix. */
    void transpose(const Genten::FacMatrix & y);

    // Compute X = X.*V, the Hadamard (elementwise) product of this matrix and V.
    void times(const Genten::FacMatrix & v);

    // Set this matrix to the Gram Matrix of V:  this = V' * V.
    void gramian(const Genten::FacMatrix & v);

    // return the index of the first entry, s, such that entry(s,c) > r.
    // assumes/requires the values entry(s,c) are nondecreasing as s increases.
    // if the assumption fails, result is undefined.
    ttb_indx firstGreaterSortedIncreasing(ttb_real r, ttb_indx c) const;

    // Compute the rank-one matrix that is the outer product of the vector v.
    /* X(i,j) = V(i) * V(j). */
    void oprod(const Genten::Array & v);

    // Compute the norms of all the columns.
    /* The norm_type indicates the type of norm.
       The optional "minval" argument sets the minimum value of each column norm. */
    void colNorms(Genten::NormType norm_type, Genten::Array & norms, ttb_real minval);

    // Scale each column by the corresponding scalar entry in s.
    /* If "inverse" is set to true, then scale by the reciprocal of the entries
     * in s. */
    void colScale(const Genten::Array & s, bool inverse);

    // see ktensor::scaleRandomElements
    void scaleRandomElements(ttb_real fraction, ttb_real scale, bool columnwise);

    // Compute the sum of all the entries (no absolute value).
    // TODO: This function really should be removed and replaced with a ktensor norm function, because that's kind of how it's used.
    ttb_real sum() const;

    // tell location of first nonfinite number (Inf or NaN) where will  be 0 if result is false,
    bool hasNonFinite(ttb_indx &where) const;

    // Permute columns in place using given column indices.
    // On return, column 0 will contain data moved from the column indicated
    // by indices[0], column 1 teh data indicated by indices[1], etc.
    void permute(const Genten::IndxArray &indices);

    //! Perform a matrix-vector multiply.
    /*!
     *  The vector can be treated as a column vector (bTranspose = false),
     *  in which case it post-multiplies the factor matrix, or as a row vector
     *  (bTranspose = true), in which case it pre-multiplies.
     *
     *  @param[in] bTranspose  True means compute x = A'x, false means x = Ax.
     *  @param[in] x  Vector of values to multiply A by.
     *  @param[out] y  Vector of result.
     */
    void multByVector(bool                bTranspose,
                      const Genten::Array &  x,
                            Genten::Array &  y) const;

    //! Solve AX = B' where B is this matrix.
    /*!
     * A must be N by N, symmetric, and nonsingular. Treat this matrix as a
     * rectangular M by N array of right-hand sides B. The method transposes
     * B, finds an LU factorization for A, solves, transposes back, and returns
     * the solution in this matrix, overwriting the original contents.
     * In Matlab notation, it performs  X = (A \ B')'.
     *
     * Since internal storage of FacMatrix data is row major, the implementation
     * is actually quite simple and avoids both logical transposes.
     *
     * Throws an exception if A is singular.
     */
    void solveTransposeRHS (const Genten::FacMatrix &  A);

    //! Multiply x elementwise by the ith row of the factor matrix, overwriting x.
    /*!
     *  Compute x = x .* A(i,:), the elementwise product of the vector x
     *  and row i.
     *
     *  @param[in,out] x  Dense vector with length matching nCols().
     *  @param[in] nRow  Row number in the factor matrix to multiply by.
     */
    void rowTimes(      Genten::Array &  x,
                  const ttb_indx      nRow) const;

    //! Multiply rows from two factor matrices elementwise.
    /*!
     *  Compute A(i,:) = A(i,:) .* B(j,:), the elementwise product of
     *  row i and row j from another factor matrix.  The result overwrites
     *  the row of this object.
     *
     *  @param[in] nRow  Row number in this factor matrix.
     *  @param[in] other  Factor matrix providing other rows.
     *  @param[in] nRowOther  Row number in the other factor matrix.
     */
    void rowTimes(const ttb_indx         nRow,
                  const Genten::FacMatrix & other,
                  const ttb_indx         nRowOther);

    //! Multiply rows from two factor matrices to get the dot product.
    /*!
     *  Compute A(i,:)' * B(j,:), where A is this factor matrix and B is
     *  the other one.
     *
     *  @param[in] nRow  Row number in this factor matrix.
     *  @param[in] other  Factor matrix providing other rows.
     *  @param[in] nRowOther  Row number in the other factor matrix.
     *  @return  Dot product of the two rows.
     */
    ttb_real rowDot(const ttb_indx         nRow,
                    const Genten::FacMatrix & other,
                    const ttb_indx         nRowOther) const;

    //! Multiply a row by a scalar, putting the result in another factor matrix.
    /*!
     *  Compute B(j,:) += s * A(i,:), where A is this factor matrix and s
     *  is a scalar constant.
     *
     *  @param[in] nRow  Row number in this factor matrix.
     *  @param[in,out] other  Factor matrix providing other rows.
     *  @param[in] nRowOther  Row number in the other factor matrix.
     *  @param[in] dScalar  Scalar factor.
     */
    void rowDScale(const ttb_indx         nRow,
                         Genten::FacMatrix & other,
                   const ttb_indx         nRowOther,
                   const ttb_real         dScalar) const;

    /** @} */


  private:

    // Data array containing the entries of the matrix.
    typedef Kokkos::View<ttb_real**,Kokkos::LayoutRight> view_type;
    view_type data;

    // ----- Private Functions -----
    // These are private to hide implementation details of the factor matrix.

    Array make_data_1d() const {
      return Array(data.span(),data.data(),true);
    }

    // Return pointer to data
    KOKKOS_INLINE_FUNCTION
    ttb_real * ptr()
    { return data.data(); }

    // Return pointer to data
    KOKKOS_INLINE_FUNCTION
    const ttb_real * ptr() const
    { return data.data(); }

public:

    // Return pointer to the ith row
    KOKKOS_INLINE_FUNCTION
    const ttb_real * rowptr(ttb_indx i) const
    { return(data.data() + i*data.stride_0()); }

    // Return pointer to the ith row
    KOKKOS_INLINE_FUNCTION
    ttb_real * rowptr(ttb_indx i)
    { return(data.data() + i*data.stride_0()); }

    KOKKOS_INLINE_FUNCTION
    view_type view() const { return data; }

private:

    //TBD why?
    friend class Sptensor;
};

}
