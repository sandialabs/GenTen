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
  @file Genten_IOtext.h
  @brief Declare methods providing I/O for Genten classes.
*/

#pragma once

#include <string>
#include <vector>
#include <ostream>
#include <memory>

#include "Genten_FacMatrix.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_Tensor.hpp"

//! Namespace for the Genten C++ project.
namespace Genten
{

  /** ----------------------------------------------------------------
     *  @name I/O Methods for Tensor (dense tensor)
     *  @{
     *  ---------------------------------------------------------------- */

  //! Read a Tensor from a text file, matching export_tensor().
  /*!
   *  <pre>
   *  The file should have three header lines followed by the entries.
   *    1st line must be the keyword 'tensor'.
   *    2nd line must provide the number of modes.
   *    3rd line must provide the sizes of all modes.
   *  </pre>
   *  Each subsequent line provides values for one element, with order
   *  determined by the storage sequence of the Tensor class.
   *  No index values are given, and rows are assumed to be dense.
   *
   *  @param[in] fName  Input filename.
   *  @param[in,out] X  Tensor resized and filled with data.
   *  @throws string    for any error.
   */
  void import_tensor (std::istream& fIn,
                      Genten::Tensor& X);
  void import_tensor (const std::string& fName,
                      Genten::Tensor& X,
                      const bool bCompressed = false);

  //! Write a Tensor to a text file, matching import_tensor().
  /*!
   *  @param[in] fName           Output filename.
   *  @param[in] X               Tensor to export.
   *  @param[in] bUseScientific  True means use "%e" format, false means "%f".
   *  @param[in] nDecimalDigits  Number of digits after the decimal point.
   *  @param[in] bCompressed     Write compressed output file if true
   *  @throws string             for any error.
   */
  void export_tensor (const std::string& fName,
                      const Genten::Tensor& X,
                      const bool bUseScientific = true,
                      const int nDecimalDigits = 15,
                      const bool bCompressed = false);

  //! Write a Tensor to an opened file stream, matching import_tensor().
  /*!
   *  See previous method for details.
   *
   *  @param[in] fOut            Stream for output.
   *                             The file is not closed by this method.
   *  @param[in] X               Tensor to export.
   *  @param[in] bUseScientific  True means use "%e" format, false means "%f".
   *  @param[in] nDecimalDigits  Number of digits after the decimal point.
   *  @throws string             for any error.
   */
  void export_tensor (std::ostream& fOut,
                      const Genten::Tensor& X,
                      const bool bUseScientific = true,
                      const int nDecimalDigits = 15);

  //! Pretty-print a tensor to an output stream.
  /*!
   *  @param[in] X     Tensor to print.
   *  @param[in] fOut  Output stream.
   *  @param[in] name  Optional name for the Tensor.
   */
  template <typename ExecSpace>
  void print_tensor (const Genten::TensorT<ExecSpace>& X,
                     std::ostream& fOut,
                     const std::string& name = "");

  /** ----------------------------------------------------------------
   *  @name I/O Methods for Sptensor (sparse tensor)
   *  @{
   *  ---------------------------------------------------------------- */

  //! Read a Sptensor from a text file, matching export_sptensor().
  /*!
   *  <pre>
   *  The file should have four header lines followed by the entries.
   *    1st line must be the keyword 'sptensor', optionally followed by the keyword 'indices-start-at-one'.
   *    2nd line must provide the number of modes.
   *    3rd line must provide the sizes of all modes.
   *    4th line must provide the number of nonzero elements.
   *  </pre>
   *  Each subsequent line provides values for one nonzero element, with
   *  indices followed by the value.  Indices start numbering at zero.
   *  The elements can be in any order.
   *
   *  @param[in] fName  Input filename.
   *  @param[in,out] X  Sptensor resized and filled with data.
   *  @throws string    for any error.
   */
  void import_sptensor (std::istream& fIn,
                        Genten::Sptensor& X,
                        const ttb_indx index_base = 0,
                        const bool verbose = false);
  void import_sptensor (const std::string& fName,
                        Genten::Sptensor& X,
                        const ttb_indx index_base = 0,
                        const bool bCompressed = false,
                        const bool verbose = false);

  //! Write a Sptensor to a text file, matching import_sptensor().
  /*!
   *  @param[in] fName           Output filename.
   *  @param[in] X               Sptensor to export.
   *  @param[in] bUseScientific  True means use "%e" format, false means "%f".
   *  @param[in] nDecimalDigits  Number of digits after the decimal point.
   *  @param[in] bStartAtZero    True if indices start at zero,
   *                             false if at one.
   *  @param[in] bCompressed     Write compressed output file if true
   *  @throws string             for any error.
   */
  void export_sptensor (const std::string& fName,
                        const Genten::Sptensor& X,
                        const bool bUseScientific = true,
                        const int nDecimalDigits = 15,
                        const bool bStartAtZero = true,
                        const bool bCompressed = false);

  //! Write a Sptensor to an opened file stream, matching import_sptensor().
  /*!
   *  See previous method for details.
   *
   *  @param[in] fOut            Stream for output.
   *                             The file is not closed by this method.
   *  @param[in] X               Sptensor to export.
   *  @param[in] bUseScientific  True means use "%e" format, false means "%f".
   *  @param[in] nDecimalDigits  Number of digits after the decimal point.
   *  @param[in] bStartAtZero    True if indices start at zero,
   *                             false if at one.
   *  @throws string             for any error.
   */
  void export_sptensor (std::ostream& fOut,
                        const Genten::Sptensor& X,
                        const bool bUseScientific = true,
                        const int nDecimalDigits = 15,
                        const bool bStartAtZero = true);

  //! Pretty-print a sparse tensor to an output stream.
  /*!
   *  @param[in] X     Sptensor to print.
   *  @param[in] fOut  Output stream.
   *  @param[in] name  Optional name for the Sptensor.
   */
  template <typename ExecSpace>
  void print_sptensor (const Genten::SptensorT<ExecSpace>& X,
                       std::ostream& fOut,
                       const std::string& name = "");

  /** @} */

  /** ----------------------------------------------------------------
   *  @name I/O Methods for FacMatrix (dense matrices)
   *  @{
   *  ---------------------------------------------------------------- */

  //! Read a factor matrix from a text file, matching export_matrix().
  /*!
   *  <pre>
   *  The file should have two header lines followed by the entries.
   *    1st line must be the keyword 'matrix' or 'facmatrix'.
   *    2nd line must provide the number of dimensions (always 2).
   *    3rd line must provide the number of rows and columns.
   *  </pre>
   *  Each subsequent line provides values for a row, with values delimited
   *  by one or more space characters.  No index values are given, and
   *  rows are assumed to be dense.
   *
   *  @param[in] fName  Input filename.
   *  @param[in,out] X  Matrix resized and filled with data.
   *  @throws string    for any error.
   */
  void import_matrix(const std::string    & fName,
                     Genten::FacMatrix & X);

  //! Read a factor matrix from an opened file stream, matching export_matrix().
  /*!
   *  See other import_matrix() method for details.
   *
   *  This method reads a factor matrix from the current position and stops
   *  reading after the last element.  The stream is allowed to have
   *  additional content after the factor matrix.
   *
   *  @param[in] fIn    File stream pointing at start of matrix data.
   *                    The file is not closed by this method.
   *  @param[in,out] X  Matrix resized and filled with data.
   *  @throws string    for any error.
   */
  void import_matrix (std::ifstream  & fIn,
                      Genten::FacMatrix & X);

  //! Write a factor matrix to a text file, matching import_matrix().
  /*!
   *  Elements are output with the default format "%0.15e".
   *
   *  @param[in] fName  Output filename.
   *  @param[in] X      Matrix to be exported.
   *  @throws string    for any error.
   */
  void export_matrix (const std::string    & fName,
                      const Genten::FacMatrix & X);

  //! Write a factor matrix to a text file, matching import_matrix().
  /*!
   *  @param[in] fName           Output filename.
   *  @param[in] X               Matrix to export.
   *  @param[in] bUseScientific  True means use "%e" format, false means "%f".
   *  @param[in] nDecimalDigits  Number of digits after the decimal point.
   *  @throws string             for any error.
   */
  void export_matrix (const std::string    & fName,
                      const Genten::FacMatrix & X,
                      const bool           & bUseScientific,
                      const int            & nDecimalDigits);

  //! Write a factor matrix to an opened file stream, matching import_matrix().
  /*!
   *  See previous method for details.
   *
   *  @param[in] fOut            File stream for output.
   *                             The file is not closed by this method.
   *  @param[in] X               Matrix to export.
   *  @param[in] bUseScientific  True means use "%e" format, false means "%f".
   *  @param[in] nDecimalDigits  Number of digits after the decimal point.
   *  @throws string             for any error.
   */
  void export_matrix (      std::ofstream  & fOut,
                            const Genten::FacMatrix & X,
                            const bool           & bUseScientific,
                            const int            & nDecimalDigits);

  //! Pretty-print a matrix to an output stream.
  /*!
   *  @param[in] X     Matrix to print.
   *  @param[in] fOut  Output stream.
   *  @param[in] name  Optional name for the matrix.
   */
  template <typename ExecSpace>
  void print_matrix (const Genten::FacMatrixT<ExecSpace>& X,
                     std::ostream& fOut,
                     const std::string& name = "");

  /** @} */

  /** ----------------------------------------------------------------
   *  @name I/O Methods for KTensor
   *  @{
   *  ---------------------------------------------------------------- */

  //! Read a Ktensor from a text file, matching export_ktensor().
  /*!
   *  <pre>
   *  The file should have four header lines followed by the entries.
   *    1st line must be the keyword 'ktensor'.
   *    2nd line must provide the number of modes.
   *    3rd line must provide the sizes of all modes.
   *    4th line must provide the number of components.
   *  </pre>
   *  Factor weights follow as a row vector, and then each factor matrix
   *  (see import_matrix() for details of their format).
   *
   *  @param[in] fName  Input filename.
   *  @param[in,out] X  Ktensor resized and filled with data.
   *  @throws string    for any error, including extra lines in the file.
   */
  void import_ktensor (const std::string  & fName,
                       Genten::Ktensor & X);

  //! Read a Ktensor from a stream, matching export_ktensor().
  /*!
   *  See other import_ktensor() method for details.
   *
   *  This method reads a ktensor from the current position and stops
   *  reading after the last element.  The stream is allowed to have
   *  additional content after the ktensor.
   *
   *  @param[in] fIn    File stream pointing at start of ktensor data.
   *                    The stream is not closed by this method.
   *  @param[in,out] X  Ktensor resized and filled with data.
   *  @throws string    for any error.
   */
  void import_ktensor (std::ifstream & fIn,
                       Genten::Ktensor  & X);

  //! Write a Ktensor to a text file, matching import_ktensor().
  /*!
   *  Elements are output with the default format "%0.15e".
   *
   *  @param[in] fName  Output filename.
   *  @param[in] X      Ktensor to be exported.
   *  @throws string    for any error.
   */
  void export_ktensor (const std::string  & fName,
                       const Genten::Ktensor & X);

  //! Write a Ktensor to a text file, matching import_ktensor().
  /*!
   *  @param[in] fName           Output filename.
   *  @param[in] X               Ktensor to export.
   *  @param[in] bUseScientific  True means use "%e" format, false means "%f".
   *  @param[in] nDecimalDigits  Number of digits after the decimal point.
   *  @throws string             for any error.
   */
  void export_ktensor (const std::string  & fName,
                       const Genten::Ktensor & X,
                       const bool         & bUseScientific,
                       const int          & nDecimalDigits);

  //! Write a Ktensor to an opened file stream, matching import_ktensor().
  /*!
   *  See previous method for details.
   *
   *  @param[in] fOut            File stream for output.
   *                             The file is not closed by this method.
   *  @param[in] X               Ktensor to export.
   *  @param[in] bUseScientific  True means use "%e" format, false means "%f".
   *  @param[in] nDecimalDigits  Number of digits after the decimal point.
   *  @throws string             for any error.
   */
  void export_ktensor (      std::ofstream & fOut,
                             const Genten::Ktensor  & X,
                             const bool          & bUseScientific,
                             const int           & nDecimalDigits);

  //! Pretty-print a Ktensor to an output stream.
  /*!
   *  @param[in] X     Ktensor to print.
   *  @param[in] fOut  Output stream.
   *  @param[in] name  Optional name for the Ktensor.
   */
  template <typename ExecSpace>
  void print_ktensor (const Genten::KtensorT<ExecSpace>& X,
                      std::ostream& fOut,
                      const std::string& name = "");

  /** @} */

  /** ----------------------------------------------------------------
   *  @name I/O Utilities
   *  @{
   *  ---------------------------------------------------------------- */

  //! Read the next line with useful content from an opened stream.
  /*!
   *  Read a line from an opened stream, dropping terminal LF (ASCII 10)
   *  and CR (ASCII 13) characters.
   *  Skip over empty lines or lines containing only white space
   *  (blank and tab).  Skip over lines containing comments that begin
   *  with "//".
   *
   *  @param[in] fIn   Input stream.
   *  @param[out] str  String variable where content is put.
   *  @return          number of lines read, including the content line,
   *                   or zero if EOF reacehd.
   */
  int getLineContent (std::istream  & fIn,
                      std::string   & str);

  //! Split a string based on white space characters into tokens.
  /*!
   *  Find tokens separated by white space (blank and tab).  Consecutive
   *  white space characters are treated as a single separator.
   *
   *  @param[in] str      String to split.
   *  @param[out] tokens  Vector of token strings.
   *  @param[in] sDelims  String of single-character delimiters,
   *                      defaults to blank and tab.
   */
  void splitStr (const std::string              &  str,
                 std::vector<std::string> &  tokens,
                 const std::string              &  sDelims = " \t");

  /** @} */

  /** ----------------------------------------------------------------
   *  @name I/O Utilities
   *  @{
   *  ---------------------------------------------------------------- */

  //! Print Sptensor
  template <typename E>
  std::ostream& operator << (std::ostream& os, const SptensorT<E>& X) {
    print_sptensor(X, os);
    return os;
  }

  //! Print Tensor
  template <typename E>
  std::ostream& operator << (std::ostream& os, const TensorT<E>& X) {
    print_tensor(X, os);
    return os;
  }

  //! Print Ktensor
  template <typename E>
  std::ostream& operator << (std::ostream& os, const KtensorT<E>& X) {
    print_ktensor(X, os);
    return os;
  }

  //! Print FacMatrix
  template <typename E>
  std::ostream& operator << (std::ostream& os, const FacMatrixT<E>& X) {
    print_matrix(X, os);
    return os;
  }

  std::pair<std::shared_ptr<std::istream>,
            std::shared_ptr<std::istream>>
  createCompressedInputFileStream(const std::string& filename);

   /** @} */

}
