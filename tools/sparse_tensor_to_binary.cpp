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

#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

bool input_is_zero_based = false;

const std::string header_string =
    R"delimiter(sptensor
ndims
size0 size1 ... size(N-1)
number non zero
ind0 ind1 ... ind(N-1) value
...
)delimiter";

struct SpTensorHeader {
  uint32_t ndims;
  std::vector<uint64_t> dim_sizes;
  std::vector<uint64_t> dim_data_sizes;
  uint64_t nnz;
  uint64_t float_data_size;
};

// I don't know if there is a smart way to do this or not
uint64_t smallestBuiltinThatHolds(uint64_t val) {
  if (val <= uint64_t(std::numeric_limits<uint16_t>::max())) {
    return 16;
  }

  if (val <= uint64_t(std::numeric_limits<uint32_t>::max())) {
    return 32;
  }

  return 64; // We didn't have a better option
}

void computeDimDataSizes(SpTensorHeader &h) {
  for (auto d : h.dim_sizes) {
    h.dim_data_sizes.emplace_back(smallestBuiltinThatHolds(d));
    std::cout << "Smallest: " << smallestBuiltinThatHolds(d) << "\n";
  }
}

std::vector<uint64_t> lineToVec(std::string const &line, int size) {
  std::vector<uint64_t> out(size);
  std::stringstream ss(line);
  for (auto i = 0; i < size; ++i) {
    ss >> out[i];
  }
  return out;
}

// Leave file at first data point
SpTensorHeader readHeader(std::ifstream &inFile, uint64_t float_data_size) {
  SpTensorHeader header;
  header.float_data_size = float_data_size;

  std::string line;
  std::getline(inFile, line);
  if (line != "sptensor") {
    throw std::invalid_argument("Input file did not start with 'sptensor'");
  }

  // This line should be the number of dimensions
  std::getline(inFile, line);
  header.ndims = std::stoi(line);

  std::getline(inFile, line);
  header.dim_sizes = lineToVec(line, header.ndims);

  std::getline(inFile, line);
  header.nnz = std::stoull(line);

  computeDimDataSizes(header);
  return header;
}

void writeIndexValue(std::ostream &outFile, SpTensorHeader const &header,
                     uint64_t value, int position) {
  uint16_t i16;
  uint32_t i32;
  switch (header.dim_data_sizes[position]) {
  case 16:
    i16 = value;
    outFile.write(reinterpret_cast<char *>(&i16), sizeof(uint16_t));
    break;
  case 32:
    i32 = value;
    outFile.write(reinterpret_cast<char *>(&i32), sizeof(uint32_t));
    break;
  default:
    outFile.write(reinterpret_cast<char *>(&value), sizeof(uint64_t));
  }
}

void writeDataValue(std::ostream &outFile, SpTensorHeader const &header,
                    double value) {
  float fp32;
  switch (header.float_data_size) {
  case 16:
    std::cout << "fp16 support not yet implemented\n";
    std::terminate();
    // fp16 = value;
    // outFile.write(reinterpret_cast<char *>(&fp16), sizeof(_Float16));
  case 32:
    fp32 = value;
    outFile.write(reinterpret_cast<char *>(&fp32), sizeof(float));
    break;
  default:
    outFile.write(reinterpret_cast<char *>(&value), sizeof(double));
  }
}

void writeRestOfTheData(std::ifstream &inFile, std::ofstream &outFile,
                        SpTensorHeader const &header) {
  const auto ndims = header.ndims;
  // This loop reads a line at a time
  while (!inFile.eof()) {
    for (auto i = 0; i < ndims; ++i) {
      uint64_t value;
      inFile >> value;
      if (!input_is_zero_based) {
        --value; // For 0 based indexing
      }
      writeIndexValue(outFile, header, value, i);
    }
    double data_value;
    inFile >> data_value;
    writeDataValue(outFile, header, data_value);
  }
}

/*
 * Takes a sparse tensor file text file and converts it to a binary file for
 * faster IO (Allows easier MPI_IO)
 *
 * The header for the text files needs to be in the form
-----------------------------------------------------------------
sptensor                   -> Type
5                          -> Number of dimensions
1605 4198 1631 4209 868131 -> Sizes of each dimension
1698825                    -> Number nonzero
1 1 1 1049 156 1.000000    -> This is the first nonzero element
...                        -> More nonzero elements
-----------------------------------------------------------------


The output file will have the following form without the newlines or -> comments
73 70 74 6e                   -> 4 char 'sptn'
ndims                         -> uint32_t
bits_for_float_type           -> uint32_t
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

/*
 * Input should be 1 argument which is the tensor file name output will be in
 * the local directory with the basename(filename).bin
 */
int main(int argc, char **argv) {
  if (argc < 2 || argc > 4) {
    std::cout << "Input is a file name to a tensor file and optionally the "
                 "size you want to store the floating point data in in bits "
                 "for now assume only {16,32,64(default)} are valid.  The "
                 "tensor file should have the following format\n"
              << header_string << "\n";
    return 1;
  }

  uint32_t float_data_size = 64;
  if (argc >= 3) {
    float_data_size = std::stoul(argv[2]);
    if (float_data_size == 16 || float_data_size == 32) {
      std::cout << "Using non-default floating point size of "
                << float_data_size << "\n";
    } else if (float_data_size != 64) {
      throw std::invalid_argument(
          "2nd argumanet for float data point size must be 16, 32, or 64.");
    }
  }

  if (argc == 4) {
    input_is_zero_based = std::stoi(argv[3]);
  }
  if (input_is_zero_based) {
    std::cout << "Assuming tensor indexing is zero based.\n";
  } else {
    std::cout << "Assuming tensor indexing is one based, will write zero based "
                 "result.\n";
  }

  std::ifstream input_file(argv[1]);
  std::string input_base = basename(argv[1]);
  std::string outfile_name = input_base + ".bin";
  std::cout << "Input filename: " << input_base << std::endl;
  std::cout << "Output filename: " << outfile_name << std::endl;

  auto header = readHeader(input_file, float_data_size);

  std::ofstream outfile(outfile_name, std::ios::binary);
  outfile.write("sptn", 4);
  outfile.write(reinterpret_cast<char *>(&header.ndims),
                sizeof(decltype(header.ndims)));
  outfile.write(reinterpret_cast<char *>(&float_data_size),
                sizeof(decltype(float_data_size)));
  for (auto i = 0; i < header.ndims; ++i) {
    auto value = header.dim_sizes[i];
    std::cout << "Writing: " << value << "\n";
    outfile.write(reinterpret_cast<char *>(&value), sizeof(uint64_t));
  }
  for (auto i = 0; i < header.ndims; ++i) {
    auto value = header.dim_data_sizes[i];
    outfile.write(reinterpret_cast<char *>(&value), sizeof(uint64_t));
  }
  outfile.write(reinterpret_cast<char *>(&(header.nnz)), sizeof(std::uint64_t));
  uint64_t total_bits = 32 /*sptn*/ + 64 /*ndims*/ + 64 /*float_data_size*/ +
                        header.ndims * 64 /* each dim size */ + 64 /* nnz */;

  uint64_t line_size = header.float_data_size;
  for (auto i = 0; i < header.ndims; ++i) {
    line_size += header.dim_data_sizes[i];
  }
  total_bits += line_size * header.nnz;

  const auto storage_in_MB = double(total_bits / 8.0) * 1e-6;
  std::cout << "Storage of output file should be: " << storage_in_MB << "MB\n";
  std::cout << "\tLine size in bytes: " << line_size / 8 << "\n";

  writeRestOfTheData(input_file, outfile, header);
  return 0;
}
