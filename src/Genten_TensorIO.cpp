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

#include <iostream>

#include "Genten_TensorIO.hpp"
#include "Genten_IOtext.hpp"
#include "Genten_AlgParams.hpp"
#include "Genten_DistContext.hpp"
#include "Genten_Tpetra.hpp"

#ifdef HAVE_DIST
#include "Genten_MPI_IO.hpp"
#endif

#ifdef HAVE_SEACAS
#include <exodusII.h>
#endif

namespace {

uint64_t smallestBuiltinThatHolds(uint64_t val) {
  if (val <= uint64_t(std::numeric_limits<uint16_t>::max())) {
    return 16;
  }
  if (val <= uint64_t(std::numeric_limits<uint32_t>::max())) {
    return 32;
  }
  return 64; // We didn't have a better option
}

}

namespace Genten {

SptnFileHeader::SptnFileHeader(const Sptensor& X,
                               const float_size_type float_size) :
  ndims(X.ndims()), float_bits(float_size), dim_lengths(ndims), dim_bits(ndims),
  nnz(X.nnz())
{
  for (nd_type n=0; n<ndims; ++n) {
    dim_lengths[n] = X.size(n);
    dim_bits[n] = smallestBuiltinThatHolds(X.size(n));
  }
}

// Initialize with user-provided values for cases when we don't have a header
SptnFileHeader::SptnFileHeader(const ptree& tree)
{
  // Parse dimensions (required)
  std::vector<int> dims;
  parse_ptree_value(tree, "dims", dims, 1, INT_MAX);
  ndims = dims.size();
  dim_lengths.resize(ndims);
  std::copy(dims.begin(), dims.end(), dim_lengths.begin());

  // Parse number of nonzeros (required)
  parse_ptree_value(tree, "nnz", nnz, 1, INT_MAX);

  // Parse bits to hold each value (optional, defaults to 64)
  float_bits = 64;
  if (tree.get_child_optional("value-bits")) {
    parse_ptree_value(tree, "value-bits", float_bits, 1, 64);
    if (float_bits != 16 && float_bits != 32 && float_bits != 64)
      Genten::error("value-bits must be one of 16, 32, or 64!");
  }

  // Parse bits to hold each subscript (optional, defaults to 32)
  std::vector<int> sub_bits(ndims, 32);
  if (tree.get_child_optional("sub-bits")) {
    parse_ptree_value(tree, "sub-bits", sub_bits, 1, 64);
    if (sub_bits.size() != ndims)
      Genten::error("sub-bits must be an array of the same length as dims!");
  }
  dim_bits.resize(ndims);
  for (nd_type n=0; n<ndims; ++n) {
    if (sub_bits[n] != 16 && sub_bits[n] != 32 && sub_bits[n] != 64)
      Genten::error("Each entry of sub-bits must be one of 16, 32, or 64!");
    dim_bits[n] = sub_bits[n];
  }
}

std::uint64_t
SptnFileHeader::bytesInDataLine() const {
  return std::accumulate(dim_bits.begin(), dim_bits.end(), float_bits) / 8;
}

std::uint64_t
SptnFileHeader::dataByteOffset() const {
  return std::accumulate(dim_bits.begin(), dim_bits.end(), 0) / 8;
}

std::uint64_t
SptnFileHeader::indByteOffset(ttb_indx ind) const {
  if (ind >= ndims) {
    throw std::out_of_range(
        "Called indByteOffset with index that was out of range\n");
  }
  auto it = dim_bits.begin();
  std::advance(it, ind);
  return std::accumulate(dim_bits.begin(), it, 0) / 8;
}

std::uint64_t
SptnFileHeader::totalBytesToRead() const {
  return bytesInDataLine() * nnz;
}

small_vector<std::uint64_t>
SptnFileHeader::getOffsetRanges(ttb_indx nranks) const {
  const ttb_indx nper_rank = nnz / nranks;
  gt_assert(nper_rank != 0);

  small_vector<std::uint64_t> out;
  out.reserve(nranks + 1);

  const std::uint64_t line_bytes = bytesInDataLine();
  std::uint64_t starting_elem = 0;
  for (ttb_indx i = 0; i < nranks; ++i) {
    out.push_back(starting_elem * line_bytes + data_starting_byte);
    starting_elem += nper_rank;
  }
  out.push_back(nnz * line_bytes + data_starting_byte);

  return out;
}

std::pair<std::uint64_t, std::uint64_t>
SptnFileHeader::getLocalOffsetRange(ttb_indx rank, ttb_indx nranks) const {
  // This is overkill and I don't care
  const auto range = getOffsetRanges(nranks);
  return {range[rank], range[rank + 1]};
};

std::vector<ttb_indx>
SptnFileHeader::getGlobalDims() const
{
  std::vector<ttb_indx> dims(ndims);
  std::copy(dim_lengths.begin(), dim_lengths.end(), dims.begin());
  return dims;
}

ttb_indx
SptnFileHeader::getGlobalNnz() const { return nnz; }

void
SptnFileHeader::readBinary(std::istream& in)
{
  std::string hi = "xxxx";
  in.read(&hi[0], 4 * sizeof(char));
  if (hi != "sptn")
    Genten::error("First 4 bytes are not sptn");

  // Number of dimensions
  in.read(reinterpret_cast<char*>(&ndims), sizeof(nd_type));

  // Floating point size
  in.read(reinterpret_cast<char*>(&float_bits), sizeof(float_size_type));

  // Size of each dimension
  dim_lengths.resize(ndims);
  for (nd_type n=0; n<ndims; ++n)
    in.read(reinterpret_cast<char*>(&dim_lengths[n]), sizeof(dim_type));

  // Subscript size
  dim_bits.resize(ndims);
  for (nd_type n=0; n<ndims; ++n)
    in.read(reinterpret_cast<char*>(&dim_bits[n]), sizeof(sub_size_type));

  // Number of nonzeros
  in.read(reinterpret_cast<char*>(&nnz), sizeof(nnz_type));
}

void
SptnFileHeader::writeBinary(std::ostream& out)
{
  out.write("sptn", 4);
  out.write(reinterpret_cast<char*>(&ndims), sizeof(nd_type));
  out.write(reinterpret_cast<char*>(&float_bits), sizeof(float_size_type));
  for (nd_type n=0; n<ndims; ++n)
    out.write(reinterpret_cast<char*>(&dim_lengths[n]), sizeof(dim_type));
  for (nd_type n=0; n<ndims; ++n)
    out.write(reinterpret_cast<char*>(&dim_bits[n]), sizeof(sub_size_type));
  out.write(reinterpret_cast<char*>(&(nnz)), sizeof(nnz_type));
}

DntnFileHeader::DntnFileHeader(const Tensor& X,
                               const float_size_type float_size) :
  ndims(X.ndims()), float_bits(float_size), dim_lengths(ndims), nnz(X.nnz())
{
  for (nd_type n=0; n<ndims; ++n)
    dim_lengths[n] = X.size(n);
}

// Initialize with user-provided values for cases when we don't have a header
DntnFileHeader::DntnFileHeader(const ptree& tree)
{
  // Parse dimensions (required)
  std::vector<ttb_indx> dims;
  parse_ptree_value(tree, "dims", dims, 1, INT_MAX);
  ndims = dims.size();
  dim_lengths.resize(ndims);
  std::copy(dims.begin(), dims.end(), dim_lengths.begin());

  // Compute number of tensor entries
  nnz = std::accumulate(dims.begin(), dims.end(), 1,
                        std::multiplies<ttb_indx>());

  // Parse bits to hold each value (optional, defaults to 64)
  float_bits = 64;
  if (tree.get_child_optional("value-bits")) {
    parse_ptree_value(tree, "value-bits", float_bits, 1, 64);
    if (float_bits != 16 && float_bits != 32 && float_bits != 64)
      Genten::error("value-bits must be one of 16, 32, or 64!");
  }
}

small_vector<std::uint64_t>
DntnFileHeader::getOffsetRanges(ttb_indx nranks) const {
  const ttb_indx nper_rank = nnz / nranks;
  gt_assert(nper_rank != 0);

  small_vector<std::uint64_t> out;
  out.reserve(nranks + 1);

  const std::uint64_t line_bytes = bytesInDataLine();
  std::uint64_t starting_elem = 0;
  for (ttb_indx i = 0; i < nranks; ++i) {
    out.push_back(starting_elem * line_bytes + data_starting_byte);
    starting_elem += nper_rank;
  }
  out.push_back(nnz * line_bytes + data_starting_byte);

  return out;
}

std::pair<std::uint64_t, std::uint64_t>
DntnFileHeader::getLocalOffsetRange(ttb_indx rank, ttb_indx nranks) const {
  // This is overkill and I don't care
  const auto range = getOffsetRanges(nranks);
  return {range[rank], range[rank + 1]};
};

std::vector<ttb_indx>
DntnFileHeader::getGlobalDims() const
{
  std::vector<ttb_indx> dims(ndims);
  std::copy(dim_lengths.begin(), dim_lengths.end(), dims.begin());
  return dims;
}

ttb_indx
DntnFileHeader::getGlobalNnz() const { return nnz; }

ttb_indx
DntnFileHeader::getGlobalElementOffset(ttb_indx rank, ttb_indx nranks) const
{
  const ttb_indx nper_rank = nnz / nranks;
  gt_assert(nper_rank != 0);
  return nper_rank*rank;
}

void
DntnFileHeader::readBinary(std::istream& in)
{
  std::string hi = "xxxx";
  in.read(&hi[0], 4 * sizeof(char));
  if (hi != "dntn")
    Genten::error("First 4 bytes are not dntn");

  // Number of dimensions
  in.read(reinterpret_cast<char*>(&ndims), sizeof(nd_type));

  // Floating point size
  in.read(reinterpret_cast<char*>(&float_bits), sizeof(float_size_type));

  // Size of each dimension
  dim_lengths.resize(ndims);
  for (nd_type n=0; n<ndims; ++n)
    in.read(reinterpret_cast<char*>(&dim_lengths[n]), sizeof(dim_type));

  // Number of nonzeros
  in.read(reinterpret_cast<char*>(&nnz), sizeof(nnz_type));
}

void
DntnFileHeader::writeBinary(std::ostream& out)
{
  out.write("dntn", 4);
  out.write(reinterpret_cast<char*>(&ndims), sizeof(nd_type));
  out.write(reinterpret_cast<char*>(&float_bits), sizeof(float_size_type));
  for (nd_type n=0; n<ndims; ++n)
    out.write(reinterpret_cast<char*>(&dim_lengths[n]), sizeof(dim_type));
  out.write(reinterpret_cast<char*>(&(nnz)), sizeof(nnz_type));
}

std::ostream&
operator<<(std::ostream& os, const SptnFileHeader& h) {
  os << "\tDimensions : " << h.ndims << "\n";
  os << "\tValue bits : " << h.float_bits << "\n";
  os << "\tMode sizes : ";
  for (auto s : h.dim_lengths) {
    os << s << " ";
  }
  os << "\n";
  os << "\tIndex bits : ";
  for (auto s : h.dim_bits) {
    os << s << " ";
  }
  os << "\n";
  os << "\tNNZ        : " << h.nnz << "\n";

  return os;
}

std::ostream&
operator<<(std::ostream& os, const DntnFileHeader& h) {
  os << "\tDimensions : " << h.ndims << "\n";
  os << "\tValue bits : " << h.float_bits << "\n";
  os << "\tMode sizes : ";
  for (auto s : h.dim_lengths) {
    os << s << " ";
  }
  os << "\n";
  os << "\tNNZ        : " << h.nnz << "\n";

  return os;
}

}

namespace {

void writeSubValue(std::ostream& outFile, const uint64_t value,
                   const uint64_t size) {
  uint16_t i16;
  uint32_t i32;
  uint64_t i64;
  switch (size) {
  case 16:
    i16 = value;
    outFile.write(reinterpret_cast<char *>(&i16), sizeof(uint16_t));
    break;
  case 32:
    i32 = value;
    outFile.write(reinterpret_cast<char *>(&i32), sizeof(uint32_t));
    break;
  default:
    i64 = value;
    outFile.write(reinterpret_cast<char *>(&i64), sizeof(uint64_t));
  }
}

uint64_t readSubValue(std::istream& inFile, const uint64_t size) {
  uint16_t i16;
  uint32_t i32;
  uint64_t i64;
  switch (size) {
  case 16:
    inFile.read(reinterpret_cast<char *>(&i16), sizeof(uint16_t));
    i64 = i16;
    break;
  case 32:
    inFile.read(reinterpret_cast<char *>(&i32), sizeof(uint32_t));
    i64 = i32;
    break;
  default:
    inFile.read(reinterpret_cast<char *>(&i64), sizeof(uint64_t));
  }
  return i64;
}

void writeDataValue(std::ostream& outFile, const double value,
                    const uint64_t size) {
  float fp32;
  double fp64;
  switch (size) {
  case 16:
    Genten::error("fp16 support not yet implemented");
    // fp16 = value;
    // outFile.write(reinterpret_cast<char*>(&fp16), sizeof(_Float16));
  case 32:
    fp32 = value;
    outFile.write(reinterpret_cast<char*>(&fp32), sizeof(float));
    break;
  default:
    fp64 = value;
    outFile.write(reinterpret_cast<char*>(&fp64), sizeof(double));
  }
}

double readDataValue(std::istream& inFile, const uint64_t size) {
  float fp32;
  double fp64 = 0.0;
  switch (size) {
  case 16:
    Genten::error("fp16 support not yet implemented");
    // inFile.read(reinterpret_cast<char *>(&fp16), sizeof(_Float16));
    // fp64 = fp16;
    break;
  case 32:
    inFile.read(reinterpret_cast<char*>(&fp32), sizeof(float));
    fp64 = fp32;
    break;
  default:
    inFile.read(reinterpret_cast<char*>(&fp64), sizeof(double));
  }
  return fp64;
}



Genten::Sptensor read_binary_sparse_tensor(const std::string& filename)
{
  std::ifstream infile(filename, std::ios::binary);
  if (!infile)
    Genten::error("Could not open input file " + filename);

  Genten::SptnFileHeader h;
  h.readBinary(infile);

  // Allocate tensor
  Genten::IndxArray sz(h.ndims);
  for (Genten::SptnFileHeader::nd_type n=0; n<h.ndims; ++n)
    sz[n] = h.dim_lengths[n];
  Genten::Sptensor x(sz, h.nnz);

  // Read nonzeros
  for (Genten::SptnFileHeader::nnz_type i=0; i<h.nnz; ++i) {
    for (Genten::SptnFileHeader::nd_type n=0; n<h.ndims; ++n)
      x.subscript(i,n) = readSubValue(infile, h.dim_bits[n]);
    x.value(i) = readDataValue(infile, h.float_bits);
  }

  return x;
}

Genten::Sptensor read_binary_sparse_tensor(const std::string& filename,
                                           const Genten::SptnFileHeader& h)
{
  std::ifstream infile(filename, std::ios::binary);
  if (!infile)
    Genten::error("Could not open input file " + filename);

  // Allocate tensor
  Genten::IndxArray sz(h.ndims);
  for (Genten::SptnFileHeader::nd_type n=0; n<h.ndims; ++n)
    sz[n] = h.dim_lengths[n];
  Genten::Sptensor x(sz, h.nnz);

  // Read nonzeros
  for (Genten::SptnFileHeader::nnz_type i=0; i<h.nnz; ++i) {
    for (Genten::SptnFileHeader::nd_type n=0; n<h.ndims; ++n)
      x.subscript(i,n) = readSubValue(infile, h.dim_bits[n]);
    x.value(i) = readDataValue(infile, h.float_bits);
  }

  return x;
}

Genten::Sptensor read_binary_dense_tensor(const std::string& filename)
{
  std::ifstream infile(filename, std::ios::binary);
  if (!infile)
    Genten::error("Could not open input file " + filename);

  Genten::DntnFileHeader h;
  h.readBinary(infile);

  // Allocate tensor
  Genten::IndxArray sz(h.ndims);
  for (Genten::DntnFileHeader::nd_type n=0; n<h.ndims; ++n)
    sz[n] = h.dim_lengths[n];
  Genten::Tensor x(sz);

  // Read onzeros
  for (Genten::DntnFileHeader::nnz_type i=0; i<h.nnz; ++i)
    x[i] = readDataValue(infile, h.float_bits);

  return x;
}

Genten::Sptensor read_binary_dense_tensor(const std::string& filename,
                                          const Genten::DntnFileHeader& h)
{
  std::ifstream infile(filename, std::ios::binary);
  if (!infile)
    Genten::error("Could not open input file " + filename);

  // Allocate tensor
  Genten::IndxArray sz(h.ndims);
  for (Genten::DntnFileHeader::nd_type n=0; n<h.ndims; ++n)
    sz[n] = h.dim_lengths[n];
  Genten::Tensor x(sz);

  // Read onzeros
  for (Genten::DntnFileHeader::nnz_type i=0; i<h.nnz; ++i)
    x[i] = readDataValue(infile, h.float_bits);

  return x;
}

Genten::Sptensor read_exodus_dense_tensor(const std::string& filename)
{
#ifdef HAVE_SEACAS
  float version;
  int CPU_word_size = 8;
  int IO_word_size  = 0;

  // open EXODUS II file
  int exoid = ex_open(filename.c_str(),  /* filename path */
                      EX_READ,           /* access mode = READ */
                      &CPU_word_size,    /* CPU word size */
                      &IO_word_size,     /* IO word size */
                      &version);         /* ExodusII library version */
  if (exoid < 0)
    Genten::error("Exodus error opeing file " + filename);

  // determine how many nodes, variables, and timesteps there are
  int   error;
  int   int_data;
  float float_data;
  char  char_data;
  error = ex_inquire(exoid, EX_INQ_NODES, &int_data, &float_data, &char_data);
  if (error != 0)
    Genten::error("Exodus error " + error);
  int num_nodes = int_data;
  error = ex_inquire(exoid, EX_INQ_NUM_NODE_VAR, &int_data, &float_data, &char_data);
  if (error != 0)
    Genten::error("Exodus error " + error);
  int num_nodal_vars = int_data;
  error = ex_inquire(exoid, EX_INQ_TIME, &int_data, &float_data, &char_data);
  if (error != 0)
    Genten::error("Exodus error " + error);
  int num_time_steps = int_data;
  // TODO: add support for element variables
  
  // collect and print the variable names
  std::vector<std::string> var_names(num_nodal_vars);
  {
    std::vector<char> temp_name(MAX_STR_LENGTH + 1);
    for(int ivar=0; ivar<num_nodal_vars; ivar++)
    {
      error = ex_get_variable_name(exoid, EX_NODAL, ivar+1, &temp_name[0]);
      if (error != 0)
        Genten::error("Exodus error " + error);
      var_names[ivar] = std::string(&temp_name[0]);
    }
    if (Genten::DistContext::rank() == 0)
    {
      std::cout << "  exodus file contains the nodal variables:";
      for(int ivar=0; ivar<num_nodal_vars; ivar++)
        std::cout << "\n    " << var_names[ivar];
      std::cout << "\n  over " << num_time_steps << " time steps\n";
      std::cout << "  on " << num_nodes << " nodes\n";
    }
  }

  // allocate tensor
  // TODO: add option to decompose into x, y, z for structured meshes
  //       add option to restrict to a plane
  Genten::IndxArray sz(3);
  sz[0] = num_nodes;
  sz[1] = num_nodal_vars;
  sz[2] = num_time_steps;
  Genten::Tensor x(sz);

  // read data into tensor
  if(CPU_word_size==4)
  {
    std::vector<float> values(num_nodes);
    for(int itime=0; itime<num_time_steps; itime++)
      for(int ivar=0; ivar<num_nodal_vars; ivar++)
      {
        error = ex_get_var(exoid, itime+1, EX_NODAL, ivar+1, 0 /*obj_id*/, num_nodes, &values[0]);
        if (error != 0)
          Genten::error("Exodus error " + error);
        for(int inode=0; inode<num_nodes; inode++)
        {
     	  int index = itime*num_nodal_vars*num_nodes + ivar*num_nodes + inode;
          x[index] = values[inode];
        }
      }
  }
  else
  {
    std::vector<double> values(num_nodes);
    for(int itime=0; itime<num_time_steps; itime++)
      for(int ivar=0; ivar<num_nodal_vars; ivar++)
      {
        error = ex_get_var(exoid, itime+1, EX_NODAL, ivar+1, 0 /*obj_id*/, num_nodes, &values[0]);
        if (error != 0)
          Genten::error("Exodus error " + error);
        for(int inode=0; inode<num_nodes; inode++)
        {
     	  int index = itime*num_nodal_vars*num_nodes + ivar*num_nodes + inode;
          x[index] = values[inode];
        }
      }
  }

  return x;

#else
  (void) filename; // fix compiler warning
  Genten::error("Cannot read exodus files without SEACAS enabled");
  return Genten::Tensor(0);
#endif
}

void write_binary_sparse_tensor(const std::string filename,
                                const Genten::Sptensor& x,
                                const bool write_header = true,
                                Genten::SptnFileHeader::float_size_type float_data_size = 64)
{
  /*
   * The output file will have the following form:
   * 73 70 74 6e                   -> 4 char 'sptn'
   * ndims                         -> uint32_t
   * bits_for_float_type           -> uint32_t
   * size0 size1 size2 size3 size4 -> ndims uint64_t
   * bits0 bits1 bits2 bits3 bits4 -> number of bits used for each index
   * number_non_zero               -> uint64_t
   * the elements depend on the size of each mode to make the file size smaller
   * we will use the smallest of 8-64 bit unsigned integer that holds all
   * the elements from the size field above, for now all floats are stored as
   * described above.  unlike the textual format we will always use zero based
   * indexing
   * 1 1 1 1049 156 1.000000 -> uint16_t uint16_t uint16_t uint16_t uint32_t
   * float_type
   */
  std::ofstream outfile(filename, std::ios::binary);
  if (!outfile)
    Genten::error("Could not open output file " + filename);

  Genten::SptnFileHeader h(x, float_data_size);
  if (write_header)
    h.writeBinary(outfile);

  for (Genten::SptnFileHeader::nnz_type i=0; i<h.nnz; ++i) {
    for (Genten::SptnFileHeader::nd_type n=0; n<h.ndims; ++n)
      writeSubValue(outfile, x.subscript(i,n), h.dim_bits[n]);
    writeDataValue(outfile, x.value(i), float_data_size);
  }
}

void write_binary_dense_tensor(const std::string filename,
                               const Genten::Tensor& x,
                               const bool write_header = true,
                               Genten::DntnFileHeader::float_size_type float_data_size = 64)
{
  /*
   * The output file will have the following form:
   * 73 70 74 6e                   -> 4 char 'dntn'
   * ndims                         -> uint32_t
   * bits_for_float_type           -> uint32_t
   * size0 size1 size2 size3 size4 -> ndims uint64_t
   * number_non_zero               -> uint64_t
   * 1.000000                      -> float_type
   */
  std::ofstream outfile(filename, std::ios::binary);
  if (!outfile)
    Genten::error("Could not open output file " + filename);

  Genten::DntnFileHeader h(x, float_data_size);
  if (write_header)
    h.writeBinary(outfile);

  for (Genten::DntnFileHeader::nnz_type i=0; i<h.nnz; ++i)
    writeDataValue(outfile, x[i], float_data_size);
}

}

namespace Genten {

template <typename ExecSpace>
TensorReader<ExecSpace>::
TensorReader(const std::string& fn,
             const ttb_indx ib,
             const bool comp,
             const ptree& tree) :
  filename(fn), index_base(ib), compressed(comp),
  is_sparse(false), is_dense(false), is_binary(false), is_text(false), is_exodus(false),
  user_header(false)
{
  if (tree.contains("format")) {
    std::string format = tree.get<std::string>("format");
    if (format == "sparse")
      is_sparse = true;
    else if (format == "dense")
      is_dense = true;
    else
      Genten::error("Invalid tensor format \"" + format + "\".  Must be \"sparse\" or \"dense\"");
  }

  if (tree.contains("file-type")) {
    std::string type = tree.get<std::string>("file-type");
    if (type == "binary")
      is_binary = true;
    else if (type == "text")
      is_text = true;
    else if (type == "exodus")
      is_exodus = true;
    else
      Genten::error("Invalid tensor file type \"" + type + "\".  Must be \"binary\" or \"text\"");
  }

  if (is_binary && tree.contains("dims") &&
      (is_dense || (is_sparse && tree.contains("nnz"))))
    user_header = true;

  if (user_header) {
    if (is_sparse) {
      sparseHeader = SptnFileHeader(tree);
      if (DistContext::rank() == 0)
        std::cout << "Reading sparse tensor with user-supplied header:\n"
                  << sparseHeader;
    }
    if (is_dense) {
      denseHeader = DntnFileHeader(tree);
      if (DistContext::rank() == 0)
        std::cout << "Reading dense tensor with user-supplied header:\n"
                  << denseHeader;
    }
  }
  else if (!is_exodus)
    queryFile();

  if (is_binary && is_sparse && index_base != 0)
    Genten::error("The binary sparse format only supports zero based indexing\n");
  if (is_binary && compressed)
    Genten::error("The binary format does not support compression\n");
}

template <typename ExecSpace>
void
TensorReader<ExecSpace>::
read()
{
  if (is_binary && is_sparse) {
    Sptensor X;
    if (user_header)
      X = read_binary_sparse_tensor(filename, sparseHeader);
    else
      X = read_binary_sparse_tensor(filename);
    X_sparse = create_mirror_view(ExecSpace(), X);
    deep_copy(X_sparse, X);
  }
  else if (is_binary && is_dense) {
    Tensor X;
    if (user_header)
      X = read_binary_dense_tensor(filename, denseHeader);
    else
      X = read_binary_dense_tensor(filename);
    X_dense = create_mirror_view(ExecSpace(), X);
    deep_copy(X_dense, X);
  }
  else if (is_text && is_sparse) {
    Sptensor X;
    Genten::import_sptensor(filename, X, index_base, compressed);
    X_sparse = create_mirror_view(ExecSpace(), X);
    deep_copy(X_sparse, X);
  }
  else if (is_text && is_dense) {
    Tensor X;
    Genten::import_tensor(filename, X, compressed);
    X_dense = create_mirror_view(ExecSpace(), X);
    deep_copy(X_dense, X);
  }
  else if (is_exodus && is_dense) {
    Tensor X;
    X = read_exodus_dense_tensor(filename);
    X_dense = create_mirror_view(ExecSpace(), X);
    deep_copy(X_dense, X);
  }
  else if (!is_sparse && !is_dense)
    Genten::error("Tensor is neither sparse nor dense, something is wrong!");
  else
    Genten::error("File is neither text nor binary, something is wrong!");
}

#ifdef HAVE_DIST
template <typename ExecSpace>
std::vector<SpDataType>
TensorReader<ExecSpace>::
parallelReadBinarySparse(std::vector<ttb_indx>& global_dims,
                         ttb_indx& nnz) const
{
  auto mpi_file = G_MPI_IO::openFile(DistContext::commWorld(), filename);
  SptnFileHeader header;
  if (user_header)
    header = sparseHeader;
  else
    header = G_MPI_IO::readSparseHeader(DistContext::commWorld(), mpi_file);
  global_dims = header.getGlobalDims();
  nnz = header.getGlobalNnz();
  return G_MPI_IO::parallelReadElements(DistContext::commWorld(),
                                        mpi_file, header);
}

template <typename ExecSpace>
std::vector<ttb_real>
TensorReader<ExecSpace>::
parallelReadBinaryDense(std::vector<ttb_indx>& global_dims,
                        ttb_indx& nnz,
                        ttb_indx& offset) const
{
  auto mpi_file = G_MPI_IO::openFile(DistContext::commWorld(), filename);
  DntnFileHeader header;
  if (user_header)
    header = denseHeader;
  else
    header = G_MPI_IO::readDenseHeader(DistContext::commWorld(), mpi_file);
  global_dims = header.getGlobalDims();
  nnz = header.getGlobalNnz();
  offset = header.getGlobalElementOffset(DistContext::rank(), DistContext::nranks());

  return G_MPI_IO::parallelReadElements(DistContext::commWorld(),
                                        mpi_file, header);
}
#endif

#if defined(HAVE_TPETRA) && defined(HAVE_SEACAS)
template <typename ExecSpace>
std::vector<ttb_real>
TensorReader<ExecSpace>::
parallelReadExodusDense(std::vector<ttb_indx>& global_dims,
                        ttb_indx& nnz,
                        ttb_indx& offset) const
{
  Teuchos::RCP<Teuchos::Comm<int>> comm = Teuchos::rcp(new Teuchos::MpiComm<int>(DistContext::commWorld()));
  int num_ranks = comm->getSize();
  int rank = comm->getRank();
  std::string this_filename = filename + "." + std::to_string(num_ranks) + "." + std::to_string(rank);

  float version;
  int CPU_word_size = 8;
  int IO_word_size  = 0;

  // open EXODUS II file
  int exoid = ex_open(this_filename.c_str(), /* filename path */
                      EX_READ,               /* access mode = READ */
                      &CPU_word_size,        /* CPU word size */
                      &IO_word_size,         /* IO word size */
                      &version);             /* ExodusII library version */
  if (exoid < 0)
    Genten::error("Exodus error opeing file " + filename);

  // determine how many nodes, variables, and timesteps there are
  int   error;
  int   int_data;
  float float_data;
  char  char_data;
  error = ex_inquire(exoid, EX_INQ_NODES, &int_data, &float_data, &char_data);
  if (error != 0)
    Genten::error("Exodus error " + error);
  int num_local_nodes = int_data;
  error = ex_inquire(exoid, EX_INQ_NUM_NODE_VAR, &int_data, &float_data, &char_data);
  if (error != 0)
    Genten::error("Exodus error " + error);
  int num_nodal_vars = int_data;
  error = ex_inquire(exoid, EX_INQ_TIME, &int_data, &float_data, &char_data);
  if (error != 0)
    Genten::error("Exodus error " + error);
  int num_time_steps = int_data;
  //error = ex_inquire(exoid, EX_INQ_NODE_MAP, &int_data, &float_data, &char_data);
  //if (error != 0)
  //  Genten::error("Exodus error " + error);
  //int num_node_maps = int_data;

  // get the LID to GID map from the exodus file
  std::vector<int> node_map_int(num_local_nodes,-1);
  error = ex_get_id_map(exoid, EX_NODE_MAP, &node_map_int[0]);

  // convert GIDs to Tpetra global type
  std::vector<tpetra_go_type> node_map(num_local_nodes,-1);
  for(std::size_t i = 0; i < node_map.size(); i++)
    node_map[i] = node_map_int[i]; 

  // build a Tpetra map
  // this includes nodes shared between ranks
  ttb_indx num_global_nodes = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
  auto ghosted_map = Teuchos::rcp(new const tpetra_map_type<ExecSpace>(num_global_nodes, &node_map[0], num_local_nodes, 0 /*index_base*/, comm));
  num_global_nodes = ghosted_map->getMaxAllGlobalIndex() + 1;

  // create a Tpetra map that divides nodes evenly between all ranks
  // this is designed to have the same distribution as the binary read
  tpetra_lo_type num_owned_nodes = num_global_nodes/comm->getSize();
  tpetra_lo_type remainder = num_global_nodes%comm->getSize();
  tpetra_go_type node_offset = rank*num_owned_nodes + std::min(rank,remainder);
  num_owned_nodes += (rank < remainder) ? 1 : 0;
  std::vector<tpetra_go_type> owned_node_map(num_owned_nodes,-1);
  for(tpetra_lo_type i = 0; i < num_owned_nodes; i++)
    owned_node_map[i] = node_offset + i;
  auto owned_map = Teuchos::rcp(new const tpetra_map_type<ExecSpace>(num_global_nodes, &owned_node_map[0], num_owned_nodes, 0 /*index_base*/, comm));

  // fill a multivector with the (owned and shared) values on this rank
  tpetra_multivector_type<ExecSpace> ghosted_multivector(ghosted_map, num_time_steps*num_nodal_vars);
  auto ghosted_data = ghosted_multivector.getLocalViewDevice(Tpetra::Access::ReadWrite);
  auto host_ghosted_data = Kokkos::create_mirror_view(DefaultHostExecutionSpace::memory_space{}, ghosted_data);
  std::vector<double> var_data(num_local_nodes);
  for(int itime = 0; itime < num_time_steps; itime++)
    for(int ivar = 0; ivar < num_nodal_vars; ivar++)
    {
      error = ex_get_var(exoid, itime+1, EX_NODAL, ivar+1, 0 /*obj_id*/, num_local_nodes, &var_data[0]);
      if (error != 0)
        Genten::error("Exodus error " + error);
      for(int inode = 0; inode < num_local_nodes; inode++)
        host_ghosted_data(inode, itime*num_nodal_vars + ivar) = var_data[inode];
    }
  Kokkos::deep_copy(ghosted_data, host_ghosted_data);

  // redistribute the data across ranks into the non-ghosted distribution
  tpetra_multivector_type<ExecSpace> owned_multivector(owned_map, num_time_steps*num_nodal_vars);
  tpetra_import_type<ExecSpace> importer(ghosted_map, owned_map);
  owned_multivector.doImport(ghosted_multivector, importer, Tpetra::INSERT);

  // set the data to return
  global_dims = std::vector<ttb_indx>(3,0);
  global_dims[2] = num_time_steps;
  global_dims[1] = num_nodal_vars;
  global_dims[0] = num_global_nodes;
  nnz = num_time_steps*num_nodal_vars*num_global_nodes;
  offset = num_time_steps*num_nodal_vars*node_offset;

  std::vector<ttb_real> result(num_owned_nodes*num_nodal_vars*num_time_steps);
  auto owned_data = owned_multivector.getLocalViewDevice(Tpetra::Access::ReadOnly);
  auto host_owned_data = Kokkos::create_mirror_view(DefaultHostExecutionSpace::memory_space{}, owned_data);
  for(int itime = 0; itime < num_time_steps; itime++)
    for(int ivar = 0; ivar < num_nodal_vars; ivar++)
      for(int inode = 0; inode < num_owned_nodes; inode++)
        result[itime*num_nodal_vars*num_owned_nodes + ivar*num_owned_nodes + inode] = host_owned_data(inode, itime*num_nodal_vars + ivar);

  return result;
}
#endif

template <typename ExecSpace>
SptnFileHeader
TensorReader<ExecSpace>::
readBinarySparseHeader() const
{
  std::ifstream infile(filename, std::ios::binary);
  if (!infile)
    Genten::error("Could not open input file " + filename);

  SptnFileHeader h;
  h.readBinary(infile);

  return h;
}

template <typename ExecSpace>
DntnFileHeader
TensorReader<ExecSpace>::
readBinaryDenseHeader() const
{
  std::ifstream infile(filename, std::ios::binary);
  if (!infile)
    Genten::error("Could not open input file " + filename);

  DntnFileHeader h;
  h.readBinary(infile);

  return h;
}

template <typename ExecSpace>
void
TensorReader<ExecSpace>::
queryFile()
{
  {
    std::ifstream file(filename, std::ios::binary);
    if (!file)
      Genten::error("Cannot open input file: " + filename);

    // First try reading the tensor as a binary file
    try {
      std::string header = "xxxx";
      file.read(&header[0], 4);
      if (header == "sptn") {
        is_sparse = true;
        is_dense  = false;
        is_binary = true;
        is_text   = false;
        return;
      }
      else if (header == "dntn") {
        is_sparse = false;
        is_dense  = true;
        is_binary = true;
        is_text   = false;
        return;
      }
    } catch (...) {}
  }

  // If that failed, try reading it as text
  try {
    std::string header;
    if (compressed) {
      auto in = Genten::createCompressedInputFileStream(filename);
      std::getline(*(in.first), header);
    }
    else {
      std::ifstream file(filename);
      std::getline(file, header);
    }
    if (header == "sptensor") {
      is_sparse = true;
      is_dense  = false;
      is_binary = false;
      is_text   = true;
      return;
    }
    else if (header == "tensor") {
      is_sparse = false;
      is_dense  = true;
      is_binary = false;
      is_text   = true;
      return;
    }
    else {
      // We support sparse tensors without a header, so try parsing the line
      // into a tuple of coordinates and a value.  stol/stod throw if the
      // conversion is invalid, so if they all succeed, we assume it is a
      // valid sparse entry.
      std::vector<std::string> tokens;
      std::stringstream ss(header);
      std::string t;
      while (std::getline(ss,t,' '))
        tokens.push_back(t);
      for (unsigned i=0; i<tokens.size()-1; ++i)
        std::stol(tokens[i]);
      std::stod(tokens[tokens.size()-1]);
      is_sparse = true;
      is_dense  = false;
      is_binary = false;
      is_text = true;
      return;
    }
  }
  catch (...) {}

  // If we got to here, it can't be read using known formats
  Genten::error("File " + filename + " cannot be read as a text or binary, sparse or dense tensor!");
}

template <typename ExecSpace>
TensorWriter<ExecSpace>::
TensorWriter(const std::string& fname,
             const bool comp) : filename(fname), compressed(comp) {}

template <typename ExecSpace>
void
TensorWriter<ExecSpace>::
writeBinary(const SptensorT<ExecSpace>& X,
            const bool write_header) const
{
  Sptensor X_host = create_mirror_view(X);
  deep_copy(X_host, X);
  write_binary_sparse_tensor(filename, X_host, write_header);
}

template <typename ExecSpace>
void
TensorWriter<ExecSpace>::
writeBinary(const TensorT<ExecSpace>& X,
            const bool write_header) const
{
  Tensor X_host = create_mirror_view(X);
  deep_copy(X_host, X);
  write_binary_dense_tensor(filename, X_host, write_header);
}

template <typename ExecSpace>
void
TensorWriter<ExecSpace>::
writeText(const SptensorT<ExecSpace>& X) const
{
  Sptensor X_host = create_mirror_view(X);
  deep_copy(X_host, X);
  export_sptensor(filename, X_host, true, 15, true, compressed);
}

template <typename ExecSpace>
void
TensorWriter<ExecSpace>::
writeText(const TensorT<ExecSpace>& X) const
{
  Tensor X_host = create_mirror_view(X);
  deep_copy(X_host, X);
  export_tensor(filename, X_host, true, 15, compressed);
}

}

#define INST_MACRO(SPACE) \
  template class Genten::TensorReader<SPACE>; \
  template class Genten::TensorWriter<SPACE>;

GENTEN_INST(INST_MACRO)
