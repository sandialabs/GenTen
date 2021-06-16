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
  if(rank == nprocs - 1){
    std::cout << header << "\n";
  }

  auto local_elems = IO::parallelReadElements(comm, file, header);
  // 
  if(rank == nprocs - 1){
    std::cout << "Last element: ";
    for(auto i = 0; i < header.ndims; ++i){
      std::cout << local_elems.back().coo[i] << " ";
    }
    std::cout << local_elems.back().val << "\n";
  }

  MPI_Comm_free(&comm);
  MPI_Finalize();
  return 0;
}
