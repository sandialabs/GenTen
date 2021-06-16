#include "Genten_MPI_IO.h"

#include <array>
#include <iostream>

namespace IO = Genten::MPI_IO;

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  if (argc != 2) {
    std::cout << "You must supply a binary file\n";
    return 1;
  }

  MPI_Comm comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);

  int nprocs, rank;
  PMPI_Comm_rank(comm, &rank);
  PMPI_Comm_size(comm, &nprocs);

  auto file = IO::openFile(comm, argv[1]);
  auto header = IO::readHeader(comm, file);

  if (rank == 1) {
    std::cout << header << "\n";
    auto range = header.getLocalOffsetRange(rank, nprocs);
    std::cout << range.first << ", " << range.second << "\n";
    auto nbytes = range.second - range.first;

    std::vector<unsigned char> data(nbytes);
    PMPI_File_read_at(file, range.first, data.data(), nbytes, MPI_BYTE,
                      MPI_STATUS_IGNORE);

    const auto ndim = header.ndims;
    std::array<std::uint64_t, 5> idx;
    auto *d = data.data() + nbytes - header.bytesInDataLine();
    for (auto i = 0; i < ndim; ++i) {
      switch (header.dim_bits[i]) {
      case 16:
        idx[i] =
            *reinterpret_cast<std::uint16_t *>(d + header.indByteOffset(i));
        break;
      case 32:
        idx[i] =
            *reinterpret_cast<std::uint32_t *>(d + header.indByteOffset(i));
        break;
      }
    }
    float f = *reinterpret_cast<float *>(d + header.dataByteOffset());
    std::cout << "Element Line: ";
    for (auto i : idx) {
      std::cout << i << " ";
    }
    std::cout << f << "\n";
  }

  PMPI_Comm_free(&comm);
  MPI_Finalize();
  return 0;
}
