#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <sstream>

struct SpHeader {
  std::uint32_t ndims;
  std::uint32_t float_bits;
  std::vector<std::uint64_t> dim_length;
  std::vector<std::uint64_t> dim_bits;
  std::uint64_t nnz;
};

std::ostream &operator<<(std::ostream &os, SpHeader const &h) {
  std::cout << "Sparse Tensor Header\n";
  std::cout << "ndims         : " << h.ndims << "\n";
  std::cout << "bits for float: " << h.float_bits << "\n";
  std::cout << "Sizes         : ";
  for (auto d : h.dim_length) {
    std::cout << d << " ";
  }
  std::cout << "\nBits          : ";
  for (auto b : h.dim_bits) {
    std::cout << b << " ";
  }
  std::cout << "\nNNZ           : " << h.nnz;

  return os;
}

SpHeader readHeader(std::istream &is) {
  {
    std::string hi = "xxxx";
    is.read(&hi[0], 4 * sizeof(char));
    if (hi != "sptn") {
      std::cout << "First 4 bytes are not sptn\n";
      std::terminate();
    }
  }

  SpHeader h;
  // Ndims
  is.read(reinterpret_cast<char *>(&h.ndims), sizeof h.ndims);

  // Num bits in float
  is.read(reinterpret_cast<char *>(&h.float_bits), sizeof h.float_bits);

  // Size of each dimension
  h.dim_length.resize(h.ndims);
  for (auto &d : h.dim_length) {
    is.read(reinterpret_cast<char *>(&d), sizeof(std::uint64_t));
  }

  // Bits for each dimension index
  h.dim_bits.resize(h.ndims);
  for (auto &b : h.dim_bits) {
    is.read(reinterpret_cast<char *>(&b), sizeof(std::uint64_t));
  }

  // NNZ
  is.read(reinterpret_cast<char *>(&h.nnz), sizeof h.nnz);

  return h;
}

std::uint64_t readNbitIntValue(std::istream &is, int bits) {
  std::uint16_t ui16;
  std::uint32_t ui32;
  std::uint64_t ui64;
  switch (bits) {
  case 16:
    is.read(reinterpret_cast<char *>(&ui16), sizeof ui16);
    ui64 = ui16;
    return ui64;
  case 32:
    is.read(reinterpret_cast<char *>(&ui32), sizeof ui32);
    ui64 = ui32;
    return ui64;
  case 64:
    is.read(reinterpret_cast<char *>(&ui64), sizeof ui64);
    return ui64;
  default:
    std::cout << "Can't read Nbit unsigned int value(" << bits
              << ") that isn't in {16, 32, 64}\n";
    std::terminate();
  }
}

double readNbitFloatValue(std::istream &is, int bits) {
  float fp;
  double dp;
  switch (bits) {
  case 32:
    is.read(reinterpret_cast<char *>(&fp), sizeof fp);
    dp = fp;
    return dp;
  case 64:
    is.read(reinterpret_cast<char *>(&dp), sizeof dp);
    return dp;
  default:
    std::cout << "Can't read Nbit float value(" << bits
              << ") that isn't in {32, 64}\n";
    std::terminate();
  }
}

void readValue(std::istream &is, std::ostream &os, SpHeader const &h) {
  const auto ndims = h.ndims;
  for(auto i = 0; i < ndims; ++i){
    os << readNbitIntValue(is, h.dim_bits[i]) << " ";
  }
  os << readNbitFloatValue(is, h.float_bits) << "\n";
}

/*
The output file will have the following form without the newlines or -> comments
73 70 74 6e                   -> 4 char 'sptn'
ndims                         -> uint32_t
bits_for_float_type          -> uint32_t
size0 size1 size2 size3 size4 -> ndims uint64_t
bits0 bits1 bits2 bits3 bits4 -> number of bits used for each index
number_non_zero               -> uint64_t
* the elements depend on the size of each mode to make the file size smaller we
* will use the smallest of uint8_t uint16_t uint32_t uint64_t that holds all
* the elements from the size field above, for now all floats are stored as
* described above.  unlike the textual format we will always use zero based
* indexing
1 1 1 1049 156 1.000000 -> uint16_t uint16_t uint16_t uint16_t uint32_t
float_type
 */
int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "You must supply a binary file\n";
    return 1;
  }

  std::ifstream input_file(argv[1], std::ios::binary);
  if (!input_file.is_open()) {
    std::cout << "Could not open input file\n";
    return 1;
  }

  auto head = readHeader(input_file);
  std::cout << head << "\n";
  
  std::stringstream ss;
  for(auto i = 0; i < head.nnz; ++i){
    readValue(input_file, ss, head);
  }

  std::ofstream test("test.txt");
  test << ss.rdbuf();

  return 0;
}
