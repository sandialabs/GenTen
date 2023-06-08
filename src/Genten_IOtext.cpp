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
  @file Genten_IOtext.cpp
  @brief Implement methods for I/O of Genten classes.
*/

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <algorithm>
#include <cctype>

#include "Genten_FacMatrix.hpp"
#include "Genten_IOtext.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_Tensor.hpp"
#include "Genten_Util.hpp"

#include "CMakeInclude.h"
#ifdef HAVE_BOOST
#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#endif

//----------------------------------------------------------------------
//  INTERNAL METHODS WITH FILE SCOPE
//----------------------------------------------------------------------

//! Get type of file to be imported and the index base.
/*!
 *  Read and verify the data type line, a single keyword plus an optional
 *  designator of the index base.  The designator is either
 *  'indices-start-at-zero' or 'indices-start-at-one' (if missing then
 *  the default is to start at zero).
 *
 *  @param[in] fIn  File stream pointing at start of data type header.
 *                  The file is not closed by this method.
 *  @param[out] sType  Data file type, which must be checked by the caller.
 *  @param[out] bStartAtZero  True if indices start with zero,
 *                            false if indices start with one.
 *  @throws string  If file read error or unknown format.
 */
static void  get_import_type (std::istream  & fIn,
                              std::string   & sType,
                              bool     & bStartAtZero)
{
  std::string  s;
  if (Genten::getLineContent(fIn,s) == 0)
  {
    Genten::error("Genten::get_import_type - cannot read from file.");
  }
  std::vector<std::string>  tokens;
  Genten::splitStr(s, tokens);
  if (tokens.size() > 2)
  {
    Genten::error("Genten::get_import_type - bad format for first line.");
  }

  bStartAtZero = true;
  if (tokens.size() == 2)
  {
    if (tokens[1].compare("indices-start-at-zero") == 0)
      bStartAtZero = true;
    else if (tokens[1].compare("indices-start-at-one") == 0)
      bStartAtZero = false;
    else
    {
      std::ostringstream  sErrMsg;
      sErrMsg << "Genten::get_import_type - 2nd word on first line"
              << " must be 'indices-start-at-zero' or 'indices-start-at-one'";
      Genten::error(sErrMsg.str());
    }
  }

  sType = tokens[0];
  return;
}

// Determine whether the given stream only contains whitespace
static bool is_white_space(std::istream& is)
{
  std::string s;
  is >> s;
  return std::all_of(s.begin(),s.end(),isspace);
}

//! Read a line and parse a single positive integer.
/*!
 *  Read and verify a line containing a specific number of positive integers.
 *
 *  @param[in] fIn        File stream pointing at start of data type header.
 *                        One line is read by this method.
 *  @param[out] naResult  Array for values read, already sized to the
 *                        expected number of integers.
 *  @throws string  If file read error.
 */
static void  read_positive_ints (      std::istream        & fIn,
                                       Genten::IndxArray & naResult,
                                       const char * const     sMsgPrefix)
{
  std::string s;
  if (Genten::getLineContent(fIn,s) == 0)
  {
    std::ostringstream  sErrMsg;
    sErrMsg << sMsgPrefix << " - cannot read line from file.";
    Genten::error(sErrMsg.str());
  }

  std::istringstream ss(s);

  for (ttb_indx  i = 0; i < naResult.size(); i++)
  {
    if (!(ss >> naResult[i]))
    {
      std::ostringstream  sErrMsg;
      sErrMsg << sMsgPrefix << " - line does not contain enough integers"
              << ", expecting " << naResult.size();
      Genten::error(sErrMsg.str());
    }
    if (naResult[i] <= 0)
    {
      std::ostringstream  sErrMsg;
      sErrMsg << sMsgPrefix << " - line must contain positive integers"
              << ", [" << i << "] is not";
      Genten::error(sErrMsg.str());
    }
  }
  if (ss.eof() == false && !is_white_space(ss))
  {
    std::ostringstream  sErrMsg;
    sErrMsg << sMsgPrefix << " - line contains too many integers"
            << " (or extra characters)"
            << ", expecting " << naResult.size();
    // Print a warning instead of an error
    std::cout << "Warning!  " << sErrMsg.str() << std::endl;
    //Genten::error(sErrMsg.str());
  }

  return;
}

//! Verify the stream is at its end.
/*!
 *  @param[in] fIn  File stream to check.
 *  @throws string  If stream is not at its end.
 */
static void  verifyEOF (      std::istream &     fIn,
                              const char * const  sMsgPrefix)
{
  std::string  s;
  if (Genten::getLineContent(fIn,s) > 0)
  {
    std::ostringstream  sErrMsg;
    sErrMsg << sMsgPrefix << " - extra lines found after last element";
    Genten::error(sErrMsg.str());
  }
  return;
}

#ifdef HAVE_BOOST
static std::pair<std::shared_ptr<boost::iostreams::filtering_stream<boost::iostreams::output>>,
                 std::shared_ptr<std::ofstream>>
createCompressedOutputFileStream(const std::string& filename)
{
  auto file = std::make_shared<std::ofstream>(filename, std::ios_base::out | std::ios_base::binary);
  if (!*file)
    Genten::error("Cannot open output file: " + filename);
  auto out = std::make_shared<boost::iostreams::filtering_stream<boost::iostreams::output> >();
  out->push(boost::iostreams::gzip_compressor());
  out->push(*file);
  return std::make_pair(out,file);
}
#else
static std::pair<std::shared_ptr<std::ostream>,
                 std::shared_ptr<std::ofstream>>
createCompressedOutputFileStream(const std::string& filename)
{
  Genten::error("Compression option requires Boost enabled.");
  return std::make_pair<std::shared_ptr<std::ostream>,std::shared_ptr<std::ofstream> >(nullptr,nullptr);
}
#endif

//----------------------------------------------------------------------
//  METHODS FOR Tensor (type "tensor")
//----------------------------------------------------------------------

void Genten::import_tensor (std::istream& fIn,
                            Genten::Tensor& X)
{
  std::string  sType;
  bool    bStartAtZero;
  get_import_type(fIn, sType, bStartAtZero);
  if (sType != "tensor")
  {
    Genten::error("Genten::import_tensor - data type header is not 'tensor'.");
  }
  //TBD support bStartAtZero

  Genten::IndxArray  naModes(1);
  read_positive_ints (fIn, naModes, "Genten::import_tensor, line 2");
  Genten::IndxArray  naSizes(naModes[0]);
  read_positive_ints (fIn, naSizes, "Genten::import_tensor, line 3");

  X = Tensor(naSizes);

  // Read the element values.
  std::string s;
  for (ttb_indx  i = 0; i < naSizes.prod(); i++)
  {
    if (getLineContent(fIn,s) == 0)
    {
      std::ostringstream  sErrMsg;
      sErrMsg << "Genten::import_tensor - error reading element " << i;
      Genten::error(sErrMsg.str());
    }
    std::istringstream ss(s);
    if (!(ss >> X[i]))
    {
      std::ostringstream  sErrMsg;
      sErrMsg << "Genten::import_tensor - error parsing element " << i;
      Genten::error(sErrMsg.str());
    }
    if (ss.eof() == false)
    {
      std::ostringstream  sErrMsg;
      sErrMsg << "Genten::import_tensor - too many values"
              << " (or extra characters) in element " << i;
      Genten::error(sErrMsg.str());
    }
  }

  verifyEOF(fIn, "Genten::import_tensor");
  return;
}

void Genten::import_tensor (const std::string& fName,
                            Genten::Tensor& X,
                            const bool bCompressed)
{
  if (bCompressed)
  {
    auto in = createCompressedInputFileStream(fName);
    import_tensor(*(in.first), X);
  }
  else
  {
    std::ifstream fIn(fName.c_str());
    if (!fIn.is_open())
    {
      Genten::error("Genten::import_tensor - cannot open input file.");
    }
    import_tensor(fIn, X);
    fIn.close();
  }
  return;
}

void Genten::export_tensor (const std::string& fName,
                            const Genten::Tensor& X,
                            const bool bUseScientific,
                            const int nDecimalDigits,
                            const bool bCompressed)
{
  if (bCompressed)
  {
    auto out = createCompressedOutputFileStream(fName);
    export_tensor(*(out.first), X, bUseScientific, nDecimalDigits);
#ifdef HAVE_BOOST
    boost::iostreams::close(*(out.first));
    out.second->close();
#endif
  }
  else
  {
    std::ofstream fOut(fName.c_str());
    if (fOut.is_open() == false)
    {
      Genten::error("Genten::export_tensor - cannot create output file.");
    }

    export_tensor(fOut, X, bUseScientific, nDecimalDigits);
    fOut.close();
  }
  return;
}

void Genten::export_tensor (std::ostream & fOut,
                            const Genten::Tensor& XX,
                            const bool bUseScientific,
                            const int nDecimalDigits)
{
  Genten::Tensor X;
  if (X.has_left_impl())
    X = XX;
  else
    X = XX.switch_layout(Genten::TensorLayout::Left);

  // Write the data type header.
  fOut << "tensor" << std::endl;

  // Write the header lines containing sizes.
  fOut << X.ndims() << std::endl;
  for (ttb_indx  i = 0; i < X.ndims(); i++)
  {
    if (i > 0)
      fOut << " ";
    fOut << X.size(i);
  }
  fOut << std::endl;

  // Apply formatting rules for elements.
  if (bUseScientific)
    fOut << std::setiosflags(std::ios::scientific);
  else
    fOut << std::fixed;
  fOut << std::setprecision(nDecimalDigits);

  // Write the elements, one per line.
  for (ttb_indx  i = 0; i < X.numel(); i++)
  {
    fOut << X[i] << std::endl;
  }

  return;
}

template <typename ExecSpace>
void Genten::print_tensor (const Genten::TensorT<ExecSpace>& X,
                           std::ostream& fOut,
                           const std::string& name)
{
  fOut << "-----------------------------------" << std::endl;
  if (name.empty())
    fOut << "tensor" << std::endl;
  else
    fOut << name << std::endl;
  fOut << "-----------------------------------" << std::endl;

  ttb_indx  nDims = X.ndims();

  fOut << "Ndims = " << nDims << std::endl;
  fOut << "Size = [ ";
  for (ttb_indx  i = 0; i < nDims; i++)
  {
    fOut << X.size(i) << " ";
  }
  fOut << "]" << std::endl;

  // Write out each element, varying indices according to ind2sub.
  Genten::IndxArray idx(nDims);
  for (ttb_indx  i = 0; i < (X.size()).prod(); i++)
  {
    X.ind2sub(idx,i);
    fOut << "X(";
    for (ttb_indx  j = 0; j < nDims; j++)
    {
      fOut << idx[j];
      if (j == (nDims - 1))
      {
        fOut << ") = ";
      }
      else
      {
        fOut << ",";
      }
    }
    fOut << X[i] << std::endl;
  }

  fOut << "-----------------------------------" << std::endl;
  return;
}

//----------------------------------------------------------------------
//  METHODS FOR Sptensor (type "sptensor")
//----------------------------------------------------------------------

void Genten::import_sptensor (std::istream& fIn,
                              Genten::Sptensor& X,
                              const ttb_indx index_base,
                              const bool verbose)
{
  // Read the first line, this tells us if we have a header, and if not,
  // how many modes there are
  std::string s;
  if (getLineContent(fIn,s) == 0)
  {
    std::ostringstream  sErrMsg;
    sErrMsg << "Genten::import_sptensor - tensor must have at "
            << "least one nonzero or a header!";
    Genten::error(sErrMsg.str());
  }
  std::vector<std::string>  tokens;
  Genten::splitStr(s, tokens);

  if (tokens.size() == 0)
  {
    std::ostringstream  sErrMsg;
    sErrMsg << "Genten::import_sptensor - invalid line:  " << s;
    Genten::error(sErrMsg.str());
  }

  ttb_indx offset = index_base;
  ttb_indx nModes = 0;
  ttb_indx nnz = 0;
  std::vector< ttb_indx> dims;
  std::vector< ttb_indx> sub_row;
  ttb_real val_row = 0.0;
  std::vector< std::vector<ttb_indx> > subs;
  std::vector< ttb_real > vals;
  bool compute_dims = true;

  // Get tensor dimensions and index base from header if we have one
  if (tokens[0] == "sptensor")
  {
    if (tokens.size() > 2)
    {
      Genten::error("Genten::get_import_type - bad format for first line.");
    }

    if (tokens.size() == 2)
    {
      if (tokens[1].compare("indices-start-at-zero") == 0)
        offset = 0;
      else if (tokens[1].compare("indices-start-at-one") == 0)
        offset = 1;
      else
      {
        std::ostringstream  sErrMsg;
        sErrMsg << "Genten::get_import_type - 2nd word on first line"
                << " must be 'indices-start-at-zero' or 'indices-start-at-one'";
        Genten::error(sErrMsg.str());
      }
    }

    // Read the number of modes, dimensions, and number of nonzeros
    Genten::IndxArray  naModes(1);
    read_positive_ints (fIn, naModes, "Genten::import_sptensor, line 2");
    Genten::IndxArray  naSizes(naModes[0]);
    read_positive_ints (fIn, naSizes, "Genten::import_sptensor, line 3");
    Genten::IndxArray  naNnz(1);
    read_positive_ints (fIn, naNnz, "Genten::import_sptensor, line 4");

    // Reserve space based on the supplied tensor dimensions
    nModes = naModes[0];
    sub_row.resize(nModes);
    dims.resize(nModes);
    for (ttb_indx i=0; i<nModes; ++i)
      dims[i] = naSizes[i];
    compute_dims = false;
    subs.reserve(naNnz[0]);
    vals.reserve(naNnz[0]);
  }

  // Otherwise this is the first nonzero and we compute the dimensions as we go
  else
  {
    nModes = tokens.size()-1;
    sub_row.resize(nModes);
    dims.resize(nModes);
    for (ttb_indx i=0; i<nModes; ++i)
    {
      sub_row[i] = std::stol(tokens[i]) - offset;
      dims[i] = sub_row[i]+1;
    }
    compute_dims = true;
    subs.push_back(sub_row);
    val_row = std::stod(tokens[nModes]);
    vals.push_back(val_row);
    ++nnz;
  }

  while (getLineContent(fIn,s))
  {
    ++nnz;
    // Don't use the above token parsing because it is too slow when
    // reading large tensors.  Instead use strtol and strtod, which already
    // handle white-space.
    char *ss = const_cast<char*>(s.c_str());
    for (ttb_indx i=0; i<nModes; ++i)
    {
      sub_row[i] = std::strtol(ss, &ss, 10) - offset;
      if (compute_dims)
        dims[i] = std::max(dims[i], sub_row[i]+1);
    }
    val_row = std::strtod(ss, &ss);
    subs.push_back(sub_row);
    vals.push_back(val_row);
  }

  verifyEOF(fIn, "Genten::import_sptensor");

  X = Sptensor(dims, vals, subs);

  if (verbose) {
    std::cout << "Read tensor with " << nnz << " nonzeros, dimensions [ ";
    for (ttb_indx i=0; i<nModes; ++i)
      std::cout << dims[i] << " ";
    std::cout << "], and starting index " << offset << std::endl;
  }
}

void Genten::import_sptensor (const std::string& fName,
                              Genten::Sptensor& X,
                              const ttb_indx index_base,
                              const bool bCompressed,
                              const bool verbose)
{
  if (bCompressed)
  {
    auto in = createCompressedInputFileStream(fName);
    import_sptensor(*(in.first), X, index_base, verbose);
  }
  else
  {
    std::ifstream fIn(fName.c_str());
    if (!fIn.is_open())
    {
      Genten::error("Genten::import_sptensor - cannot open input file.");
    }
    import_sptensor(fIn, X, index_base, verbose);
    fIn.close();
  }
}

void Genten::export_sptensor (const std::string& fName,
                              const Genten::Sptensor& X,
                              const bool bUseScientific,
                              const int nDecimalDigits,
                              const bool bStartAtZero,
                              const bool bCompressed)
{
  if (bCompressed)
  {
    auto out = createCompressedOutputFileStream(fName);
    export_sptensor(*(out.first), X, bUseScientific, nDecimalDigits,
                    bStartAtZero);
#ifdef HAVE_BOOST
    boost::iostreams::close(*(out.first));
    out.second->close();
#endif
  }
  else
  {
    std::ofstream fOut(fName.c_str());
    if (fOut.is_open() == false)
    {
      Genten::error("Genten::export_sptensor - cannot create output file.");
    }

    export_sptensor(fOut, X, bUseScientific, nDecimalDigits, bStartAtZero);
    fOut.close();
  }
  return;
}

void Genten::export_sptensor (std::ostream& fOut,
                              const Genten::Sptensor& X,
                              const bool bUseScientific,
                              const int nDecimalDigits,
                              const bool bStartAtZero)
{
  // Write the data type header.
  if (bStartAtZero)
    fOut << "sptensor" << std::endl;
  else
    fOut << "sptensor indices-start-at-one" << std::endl;

  // Write the header lines containing sizes.
  fOut << X.ndims() << std::endl;
  for (ttb_indx  i = 0; i < X.ndims(); i++)
  {
    if (i > 0)
      fOut << " ";
    fOut << X.size(i);
  }
  fOut << std::endl;
  fOut << X.nnz() << std::endl;

  // Apply formatting rules for elements.
  if (bUseScientific)
    fOut << std::setiosflags(std::ios::scientific);
  else
    fOut << std::fixed;
  fOut << std::setprecision(nDecimalDigits);

  // Write the nonzero elements, one per line.
  for (ttb_indx  i = 0; i < X.nnz(); i++)
  {
    for (ttb_indx  j = 0; j < X.ndims(); j++)
    {
      if (bStartAtZero)
        fOut << X.subscript(i,j) << " ";
      else
        fOut << X.subscript(i,j) + 1 << " ";
    }
    fOut << X.value(i) << std::endl;
  }

  return;
}

template <typename ExecSpace>
void Genten::print_sptensor (const Genten::SptensorT<ExecSpace>& X,
                             std::ostream& fOut,
                             const std::string& name)
{
  fOut << "-----------------------------------" << std::endl;
  if (name.empty())
    fOut << "sptensor" << std::endl;
  else
    fOut << name << std::endl;
  fOut << "-----------------------------------" << std::endl;

  ttb_indx  nDims = X.ndims();

  fOut << "Ndims = " << nDims << std::endl;
  fOut << "Size = [ ";
  for (ttb_indx  i = 0; i < nDims; i++)
  {
    fOut << X.size(i) << " ";
  }
  fOut << "]" << std::endl;

  fOut << "NNZ = " << X.nnz() << std::endl;

  // Write out each element.
  for (ttb_indx  i = 0; i < X.nnz(); i++)
  {
    fOut << "X(";
    for (ttb_indx  j = 0; j < nDims; j++)
    {
      fOut << X.subscript(i,j);
      if (j == (nDims - 1))
      {
        fOut << ") = ";
      }
      else
      {
        fOut << ",";
      }
    }
    fOut << X.value(i) << std::endl;
  }

  fOut << "-----------------------------------" << std::endl;
  return;
}


//----------------------------------------------------------------------
//  METHODS FOR FacMatrix (type "matrix")
//----------------------------------------------------------------------

void Genten::import_matrix (const std::string         & fName,
                            Genten::FacMatrix & X)
{
  std::ifstream fIn(fName.c_str());
  import_matrix(fIn, X);

  verifyEOF(fIn, "Genten::import_matrix");
  fIn.close();
  return;
}

void Genten::import_matrix (std::ifstream       & fIn,
                            Genten::FacMatrix & X)
{
  if (fIn.is_open() == false)
  {
    Genten::error("Genten::import_matrix - cannot open input file.");
  }

  std::string  sType;
  bool    bStartAtZero;
  get_import_type(fIn, sType, bStartAtZero);
  if ((sType != "facmatrix") && (sType != "matrix"))
  {
    Genten::error("Genten::import_matrix - data type header is not 'matrix'.");
  }
  //TBD support bStartAtZero

  // Process second and third lines.
  Genten::IndxArray  naModes(1);
  read_positive_ints (fIn, naModes, "Genten::import_matrix, number of dimensions should be 2");
  if (naModes[0] != 2)
  {
    Genten::error("Genten::import_matrix - illegal number of dimensions");
  }
  Genten::IndxArray  naTmp(2);
  read_positive_ints (fIn, naTmp, "Genten::import_matrix, line 3");
  ttb_indx nRows = naTmp[0];
  ttb_indx nCols = naTmp[1];

  X = Genten::FacMatrix(nRows, nCols);

  // Read the remaining lines, expecting one row of values per line.
  // Extra lines are ignored (allowing for multiple matrices in one file).
  std::string s;
  for (ttb_indx  i = 0; i < nRows; i++)
  {
    if (getLineContent(fIn,s) == 0)
    {
      std::ostringstream  sErrMsg;
      sErrMsg << "Genten::import_matrix - error reading row " << i
              << " of " << nRows;
      Genten::error(sErrMsg.str());
    }
    std::istringstream ss(s);
    for (ttb_indx  j = 0; j < nCols; j++)
    {
      if (!(ss >> X.entry(i,j)))
      {
        std::ostringstream  sErrMsg;
        sErrMsg << "Genten::import_matrix - error reading column " << j
                << " of row " << i << " (out of " << nRows << " rows)";
        Genten::error(sErrMsg.str());
      }
    }
    if (ss.eof() == false && !is_white_space(ss))
    {
      std::ostringstream  sErrMsg;
      sErrMsg << "Genten::import_matrix - too many values"
              << " (or extra characters) in row " << i;
      Genten::error(sErrMsg.str());
    }
  }

  return;
}

void Genten::export_matrix (const std::string    & fName,
                            const Genten::FacMatrix & X)
{
  export_matrix (fName, X, true, 15);
  return;
}

void Genten::export_matrix (const std::string    & fName,
                            const Genten::FacMatrix & X,
                            const bool           & bUseScientific,
                            const int            & nDecimalDigits)
{
  std::ofstream fOut(fName.c_str());
  if (fOut.is_open() == false)
  {
    Genten::error("Genten::export_matrix - cannot create output file.");
  }

  export_matrix(fOut, X, bUseScientific, nDecimalDigits);
  fOut.close();
  return;
}

void Genten::export_matrix (      std::ofstream  & fOut,
                                  const Genten::FacMatrix & X,
                                  const bool           & bUseScientific,
                                  const int            & nDecimalDigits)
{
  if (fOut.is_open() == false)
  {
    Genten::error("Genten::export_matrix - cannot create output file.");
  }

  // Write the data type header.
  fOut << "matrix" << std::endl;

  // Write the number of modes and size of each mode,
  // consistent with Matlab Genten.
  fOut << "2" << std::endl;
  fOut << X.nRows() << " " << X.nCols() << std::endl;

  // Apply formatting rules for elements.
  if (bUseScientific)
    fOut << std::setiosflags(std::ios::scientific);
  else
    fOut << std::fixed;
  fOut << std::setprecision(nDecimalDigits);

  // Write elements for each row on one line.
  for (ttb_indx  i = 0; i < X.nRows(); i++)
  {
    for (ttb_indx  j = 0; j < X.nCols(); j++)
    {
      if (j > 0)
        fOut << " ";
      fOut << X.entry(i,j);
    }
    fOut << std::endl;
  }

  return;
}

template <typename ExecSpace>
void Genten::print_matrix (const Genten::FacMatrixT<ExecSpace> & X,
                           std::ostream& fOut,
                           const std::string& name)
{
  fOut << "-----------------------------------" << std::endl;
  if (name.empty())
    fOut << "matrix" << std::endl;
  else
    fOut << name << std::endl;
  fOut << "-----------------------------------" << std::endl;

  fOut << "Size = [ " << X.nRows() << " " << X.nCols() << " ]" << std::endl;

  for (ttb_indx  j = 0; j < X.nCols(); j++)
  {
    for (ttb_indx  i = 0; i < X.nRows(); i++)
    {
      fOut << "X(" << i << "," << j << ") = " << X.entry(i,j) << std::endl;
    }
  }

  fOut << "-----------------------------------" << std::endl;
  return;
}


//----------------------------------------------------------------------
//  METHODS FOR Ktensor (type "ktensor")
//----------------------------------------------------------------------

void Genten::import_ktensor (const std::string  & fName,
                             Genten::Ktensor & X)
{
  std::ifstream fIn(fName.c_str());
  import_ktensor (fIn, X);

  verifyEOF(fIn, "Genten::import_ktensor");
  fIn.close();
  return;
}

void Genten::import_ktensor (std::ifstream & fIn,
                             Genten::Ktensor  & X)
{
  if (fIn.is_open() == false)
  {
    Genten::error("Genten::import_ktensor - cannot open input file.");
  }

  std::string  sType;
  bool    bStartAtZero;
  get_import_type(fIn, sType, bStartAtZero);
  if (sType != "ktensor")
  {
    Genten::error("Genten::import_ktensor - data type header is not 'ktensor'.");
  }
  //TBD support bStartAtZero

  Genten::IndxArray  naModes(1);
  read_positive_ints (fIn, naModes, "Genten::import_ktensor, line 2");
  Genten::IndxArray  naSizes(naModes[0]);
  read_positive_ints (fIn, naSizes, "Genten::import_ktensor, line 3");
  Genten::IndxArray  naComps(1);
  read_positive_ints (fIn, naComps, "Genten::import_ktensor, line 4");

  X = Genten::Ktensor(naComps[0], naModes[0]);

  // Read the factor weights.
  std::string  s;
  if (getLineContent(fIn,s) == 0)
  {
    Genten::error("Genten::import_ktensor - cannot read line with weights");
  }
  Genten::Array  daWeights(naComps[0]);
  std::istringstream ss(s);
  for (ttb_indx  i = 0; i < naComps[0]; i++)
  {
    if (!(ss >> daWeights[i]))
    {
      std::ostringstream  sErrMsg;
      sErrMsg << "Genten::import_ktensor - error reading weight " << i;
      Genten::error(sErrMsg.str());
    }
    if (daWeights[i] < 0.0)
    {
      Genten::error("Genten::import_ktensor - factor weight cannot be negative");
    }
  }
  if (ss.eof() == false && !is_white_space(ss))
  {
    std::ostringstream  sErrMsg;
    sErrMsg << "Genten::import_ktensor - too many values"
            << " (or extra characters) in weights vector";
    Genten::error(sErrMsg.str());
  }
  X.setWeights(daWeights);

  // Read the factor matrices.
  for (ttb_indx  i = 0; i < naModes[0]; i++)
  {
    Genten::FacMatrix  nextFactor;
    import_matrix (fIn, nextFactor);
    if (   (nextFactor.nRows() != naSizes[i])
           || (nextFactor.nCols() != naComps[0]) )
    {
      std::ostringstream  sErrMsg;
      sErrMsg << "Genten::import_ktensor - factor matrix " << i
              << " is not the correct size"
              << ", expecting " << naSizes[i] << " by " << naComps[0];
      Genten::error(sErrMsg.str());
    }
    X.set_factor(i, nextFactor);
  }

  return;
}

void Genten::export_ktensor (const std::string  & fName,
                             const Genten::Ktensor & X)
{
  export_ktensor (fName, X, true, 15);
  return;
}

void Genten::export_ktensor (const std::string  & fName,
                             const Genten::Ktensor & X,
                             const bool         & bUseScientific,
                             const int          & nDecimalDigits)
{
  std::ofstream fOut(fName.c_str());
  if (fOut.is_open() == false)
  {
    Genten::error("Genten::export_ktensor - cannot create output file.");
  }

  export_ktensor(fOut, X, bUseScientific, nDecimalDigits);
  fOut.close();
  return;
}

void Genten::export_ktensor (      std::ofstream & fOut,
                                   const Genten::Ktensor  & X,
                                   const bool          & bUseScientific,
                                   const int           & nDecimalDigits)
{
  if (fOut.is_open() == false)
  {
    Genten::error("Genten::export_ktensor - cannot create output file.");
  }

  // Write the data type header.
  fOut << "ktensor" << std::endl;

  // Write the header lines containing sizes.
  fOut << X.ndims() << std::endl;
  for (ttb_indx  i = 0; i < X.ndims(); i++)
  {
    if (i > 0)
      fOut << " ";
    fOut << X[i].nRows();
  }
  fOut << std::endl;
  fOut << X.ncomponents() << std::endl;

  // Apply formatting rules for elements.
  if (bUseScientific)
    fOut << std::setiosflags(std::ios::scientific);
  else
    fOut << std::fixed;
  fOut << std::setprecision(nDecimalDigits);

  // Write the weights on one line.
  for (ttb_indx  i = 0; i < X.ncomponents(); i++)
  {
    if (i > 0)
      fOut << " ";
    fOut << X.weights(i);
  }
  fOut << std::endl;

  // Write the factor matrices.
  for (ttb_indx  i = 0; i < X.ndims(); i++)
  {
    export_matrix(fOut, X[i], bUseScientific, nDecimalDigits);
  }

  return;
}

template <typename ExecSpace>
void Genten::print_ktensor (const Genten::KtensorT<ExecSpace>& X,
                            std::ostream& fOut,
                            const std::string& name)
{
  fOut << "-----------------------------------" << std::endl;
  if (name.empty())
    fOut << "ktensor" << std::endl;
  else
    fOut << name << std::endl;
  fOut << "-----------------------------------" << std::endl;

  ttb_indx nd = X.ndims();
  ttb_indx nc = X.ncomponents();
  fOut << "Ndims = " << nd <<"    Ncomps = " << nc << std::endl;

  fOut << "Size = [ ";
  for (ttb_indx  k = 0; k < nd; k++)
  {
    fOut << X[k].nRows() << ' ';
  }
  fOut << "]" << std::endl;

  fOut << "Weights = [ ";
  for (ttb_indx  k = 0; k < nc; k++)
  {
    fOut << X.weights(k) << ' ';
  }
  fOut << "]" << std::endl;

  for (ttb_indx  k = 0; k < nd; k++)
  {
    fOut << "Factor " << k << std::endl;
    for (ttb_indx  j = 0; j < X[k].nCols(); j++)
    {
      for (ttb_indx  i = 0; i < X[k].nRows(); i++)
      {
        fOut << "f" << k << "(" << i << "," << j << ") = "
             << X[k].entry(i,j) << std::endl;
      }
    }
  }

  fOut << "-----------------------------------" << std::endl;
  return;
}

//----------------------------------------------------------------------
//  UTILITY METHODS
//----------------------------------------------------------------------

//! Read the next line with useful content from an opened stream.
/*!
 *  Read a line from an opened stream, dropping the '\n' character.
 *  Skip over lines containing comments that begin with "//".
 *  For platform independence, check if the last character is '\r'
 *  (Windows text file) and remove if it is.
 *
 *  For better read performance, this no longer checks for blank lines,
 *  which can easily double the cost of reading the tensor.
 *  If that is important, we can maybe make it optional.
 *
 *  @param[in]  fIn  Input stream.
 *  @param[out] str  String variable where content is put.
 *  @return          number of lines read, including the content line,
 *                   or zero if EOF reacehd.
 */
int Genten::getLineContent (std::istream  & fIn,
                            std::string   & str)
{
  int  nNumLines = 0;
  while (true)
  {
    getline(fIn, str);
    if (fIn.eof() == true)
    {
      str = "";
      return( 0 );
    }
    nNumLines++;

    // Remove end-of-line character(s).
    int  nLast = (int) str.size();
    if (str[nLast-1] == '\r')
      str.erase(nLast-1, 1);

    // Remove comment lines
    if ((str[0] != '/') || (str[1] != '/'))
      return( nNumLines );
  }
}

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
void Genten::splitStr (const std::string         &  str,
                       std::vector<std::string> &  tokens,
                       const std::string         &  sDelims)
{
  size_t  nStart;
  size_t  nEnd = 0;
  while (nEnd < str.size())
  {
    nStart = nEnd;
    // Skip past any initial delimiter characters.
    while (   (nStart < str.size())
              && (sDelims.find(str[nStart]) != std::string::npos))
    {
      nStart++;
    }
    nEnd = nStart;

    // Find the end of the token.
    while (   (nEnd < str.size())
              && (sDelims.find(str[nEnd]) == std::string::npos))
    {
      nEnd++;
    }

    // Save the token if nonempty.
    if (nEnd - nStart != 0)
    {
      tokens.push_back (std::string(str, nStart, nEnd - nStart));
    }
  }

  return;
}

std::pair<std::shared_ptr<std::istream>,std::shared_ptr<std::istream>>
Genten::createCompressedInputFileStream(const std::string& filename)
{
#ifdef HAVE_BOOST
  auto file = std::make_shared<std::ifstream>(filename, std::ios_base::in | std::ios_base::binary);
  if (!*file)
    Genten::error("Cannot open input file: " + filename);
  auto in = std::make_shared<boost::iostreams::filtering_stream<boost::iostreams::input> >();
  in->push(boost::iostreams::gzip_decompressor());
  in->push(*file);
  return std::make_pair<std::shared_ptr<std::istream>,std::shared_ptr<std::istream> >(in,file);
#else
  Genten::error("Compression option requires Boost enabled.");
  return std::make_pair<std::shared_ptr<std::istream>,std::shared_ptr<std::istream> >(nullptr,nullptr);
#endif
}

#define INST_MACRO(SPACE)                                               \
  template void print_tensor (const Genten::TensorT<SPACE>& X,          \
                              std::ostream& fOut,                       \
                              const std::string& name);                 \
  template void print_sptensor (const Genten::SptensorT<SPACE>& X,      \
                                std::ostream& fOut,                     \
                                const std::string& name);               \
  template void print_ktensor (const Genten::KtensorT<SPACE>& X,        \
                               std::ostream& fOut,                      \
                               const std::string& name);                \
  template void print_matrix (const Genten::FacMatrixT<SPACE>& X,       \
                              std::ostream& fOut,                       \
                              const std::string& name);

GENTEN_INST(INST_MACRO)
