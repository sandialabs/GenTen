#include <Genten_SpTn_Util.h>

#include <iostream>
#include <cassert>
#include <numeric>
#include <stdexcept>

namespace Genten {
namespace MPI_IO {

std::uint64_t SptnFileHeader::bytesInDataLine() const {
  return std::accumulate(dim_bits.begin(), dim_bits.end(), float_bits) / 8;
}

std::uint64_t SptnFileHeader::dataByteOffset() const {
  return std::accumulate(dim_bits.begin(), dim_bits.end(), 0) / 8;
}

std::uint64_t SptnFileHeader::indByteOffset(int ind) const {
  if (ind >= ndims) {
    throw std::out_of_range(
        "Called indByteOffset with index that was out of range\n");
  }
  auto it = dim_bits.begin();
  std::advance(it, ind);
  return std::accumulate(dim_bits.begin(), it, 0) / 8;
}

std::uint64_t SptnFileHeader::totalBytesToRead() const {
  return bytesInDataLine() * nnz;
}

small_vector<std::uint64_t> SptnFileHeader::getOffsetRanges(int nranks) const {
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

TensorInfo SptnFileHeader::toTensorInfo() const {
  TensorInfo Ti;

  Ti.nnz = nnz;
  Ti.dim_sizes.resize(ndims);
  std::copy(dim_lengths.begin(), dim_lengths.end(), Ti.dim_sizes.begin());

  return Ti;
}

std::ostream &operator<<(std::ostream &os, SptnFileHeader const &h) {
  os << "Sparse Tensor Info :\n";
  os << "\tDimensions : " << h.ndims << "\n";
  os << "\tFloat bits : " << h.float_bits << "\n";
  os << "\tSizes      : ";
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
  os << "\tData Byte  : " << h.data_starting_byte;

  return os;
}
} // namespace MPI_IO
} // namespace Genten
