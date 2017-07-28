module load intel/compilers/17.0.098
module load cmake/3.4.3
module load boost/1.59.0/openmpi/1.10.2/intel/17.0.042
export OMP_NUM_THREADS=64
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
