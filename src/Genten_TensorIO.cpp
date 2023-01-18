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

#include "Genten_TensorIO.hpp"
#include "Genten_IOtext.hpp"
#include "Genten_MPI_IO.hpp"

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
  for (auto n=0; n<ndims; ++n) {
    dim_lengths[n] = X.size(n);
    dim_bits[n] = smallestBuiltinThatHolds(X.size(n));
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
SptnFileHeader::indByteOffset(int ind) const {
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
SptnFileHeader::getOffsetRanges(int nranks) const {
  const auto nper_rank = nnz / nranks;
  assert(nper_rank != 0);

  small_vector<std::uint64_t> out;
  out.reserve(nranks + 1);

  const auto line_bytes = bytesInDataLine();
  std::uint64_t starting_elem = 0;
  for (auto i = 0; i < nranks; ++i) {
    out.push_back(starting_elem * line_bytes + data_starting_byte);
    starting_elem += nper_rank;
  }
  out.push_back(nnz * line_bytes + data_starting_byte);

  return out;
}

std::pair<std::uint64_t, std::uint64_t>
SptnFileHeader::getLocalOffsetRange(int rank, int nranks) const {
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
  for (auto n=0; n<ndims; ++n)
    in.read(reinterpret_cast<char*>(&dim_lengths[n]), sizeof(dim_type));

  // Subscript size
  dim_bits.resize(ndims);
  for (auto n=0; n<ndims; ++n)
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
  for (auto n=0; n<ndims; ++n)
    out.write(reinterpret_cast<char*>(&dim_lengths[n]), sizeof(dim_type));
  for (auto n=0; n<ndims; ++n)
    out.write(reinterpret_cast<char*>(&dim_bits[n]), sizeof(sub_size_type));
  out.write(reinterpret_cast<char*>(&(nnz)), sizeof(nnz_type));
}

DntnFileHeader::DntnFileHeader(const Tensor& X,
                               const float_size_type float_size) :
  ndims(X.ndims()), float_bits(float_size), dim_lengths(ndims), nnz(X.nnz())
{
  for (auto n=0; n<ndims; ++n)
    dim_lengths[n] = X.size(n);
}

small_vector<std::uint64_t>
DntnFileHeader::getOffsetRanges(int nranks) const {
  const auto nper_rank = nnz / nranks;
  assert(nper_rank != 0);

  small_vector<std::uint64_t> out;
  out.reserve(nranks + 1);

  const auto line_bytes = bytesInDataLine();
  std::uint64_t starting_elem = 0;
  for (auto i = 0; i < nranks; ++i) {
    out.push_back(starting_elem * line_bytes + data_starting_byte);
    starting_elem += nper_rank;
  }
  out.push_back(nnz * line_bytes + data_starting_byte);

  return out;
}

std::pair<std::uint64_t, std::uint64_t>
DntnFileHeader::getLocalOffsetRange(int rank, int nranks) const {
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
  float_size_type float_data_size;
  in.read(reinterpret_cast<char*>(&float_bits), sizeof(float_size_type));

  // Size of each dimension
  dim_lengths.resize(ndims);
  for (auto n=0; n<ndims; ++n)
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
  for (auto n=0; n<ndims; ++n)
    out.write(reinterpret_cast<char*>(&dim_lengths[n]), sizeof(dim_type));
  out.write(reinterpret_cast<char*>(&(nnz)), sizeof(nnz_type));
}

// std::ostream
// &operator<<(std::ostream &os, SptnFileHeader const &h) {
//   os << "Sparse Tensor Info :\n";
//   os << "\tDimensions : " << h.ndims << "\n";
//   os << "\tFloat bits : " << h.float_bits << "\n";
//   os << "\tSizes      : ";
//   for (auto s : h.dim_lengths) {
//     os << s << " ";
//   }
//   os << "\n";
//   os << "\tIndex bits : ";
//   for (auto s : h.dim_bits) {
//     os << s << " ";
//   }
//   os << "\n";
//   os << "\tNNZ        : " << h.nnz << "\n";
//   os << "\tData Byte  : " << h.data_starting_byte;

//   return os;
// }

// std::ostream
// &operator<<(std::ostream &os, DntnFileHeader const &h) {
//   os << "Dense Tensor Info :\n";
//   os << "\tDimensions : " << h.ndims << "\n";
//   os << "\tFloat bits : " << h.float_bits << "\n";
//   os << "\tSizes      : ";
//   for (auto s : h.dim_lengths) {
//     os << s << " ";
//   }
//   os << "\n";
//   os << "\tNNZ        : " << h.nnz << "\n";
//   os << "\tData Byte  : " << h.data_starting_byte;

//   return os;
// }

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
  double fp64;
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
  for (auto n=0; n<h.ndims; ++n)
    sz[n] = h.dim_lengths[n];
  Genten::Sptensor x(sz, h.nnz);

  // Read nonzeros
  for (auto i=0; i<h.nnz; ++i) {
    for (auto n=0; n<h.ndims; ++n)
      x.subscript(i,n) = readSubValue(infile, h.dim_bits[n]);
    x.value(i) = readDataValue(infile, h.float_bits);
  }

  return x;
}

Genten::Sptensor read_binary_dense_tensor(const std::string filename)
{
  std::ifstream infile(filename, std::ios::binary);
  if (!infile)
    Genten::error("Could not open input file " + filename);

  Genten::DntnFileHeader h;
  h.readBinary(infile);

  // Allocate tensor
  Genten::IndxArray sz(h.ndims);
  for (auto n=0; n<h.ndims; ++n)
    sz[n] = h.dim_lengths[n];
  Genten::Tensor x(sz);

  // Read onzeros
  for (auto i=0; i<h.nnz; ++i)
    x[i] = readDataValue(infile, h.float_bits);

  return x;
}

void write_binary_sparse_tensor(const std::string filename,
                                const Genten::Sptensor& x,
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
  h.writeBinary(outfile);

  for (auto i=0; i<h.nnz; ++i) {
    for (auto n=0; n<h.ndims; ++n)
      writeSubValue(outfile, x.subscript(i,n), h.dim_bits[n]);
    writeDataValue(outfile, x.value(i), float_data_size);
  }
}

void write_binary_dense_tensor(const std::string filename,
                               const Genten::Tensor& x,
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
  h.writeBinary(outfile);
  for (auto i=0; i<h.nnz; ++i)
    writeDataValue(outfile, x[i], float_data_size);
}

}

namespace Genten {

template <typename ExecSpace>
TensorReader<ExecSpace>::
TensorReader(const std::string& fn,
             const ttb_indx ib,
             const bool comp) :
  filename (fn), index_base(ib), compressed(comp),
  is_sparse(false), is_dense(false), is_binary(false), is_text(false)
{
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
    Sptensor X = read_binary_sparse_tensor(filename);
    X_sparse = create_mirror_view(ExecSpace(), X);
    deep_copy(X_sparse, X);
  }
  else if (is_binary && is_dense) {
    Tensor X = read_binary_dense_tensor(filename);
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
  auto header = G_MPI_IO::readSparseHeader(DistContext::commWorld(),
                                             mpi_file);
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
  auto header = G_MPI_IO::readDenseHeader(DistContext::commWorld(),
                                          mpi_file);
  global_dims = header.getGlobalDims();
  nnz = header.getGlobalNnz();
  offset = header.getLocalOffsetRange(DistContext::rank(),
                                      DistContext::nranks()).first;
  return G_MPI_IO::parallelReadElements(DistContext::commWorld(),
                                        mpi_file, header);
}
#endif

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
        is_binary = true;
        return;
      }
      else if (header == "dntn") {
        is_dense = true;
        is_binary = true;
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
      is_binary = true;
      return;
    }
    else if (header == "tensor") {
      is_dense = true;
      is_text = true;
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
      for (auto i=0; i<tokens.size()-1; ++i)
        std::stol(tokens[i]);
      std::stod(tokens[tokens.size()-1]);
      is_sparse = true;
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
writeBinary(const SptensorT<ExecSpace>& X) const
{
  Sptensor X_host = create_mirror_view(X);
  deep_copy(X_host, X);
  write_binary_sparse_tensor(filename, X_host);
}

template <typename ExecSpace>
void
TensorWriter<ExecSpace>::
writeBinary(const TensorT<ExecSpace>& X) const
{
  Tensor X_host = create_mirror_view(X);
  deep_copy(X_host, X);
  write_binary_dense_tensor(filename, X_host);
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
