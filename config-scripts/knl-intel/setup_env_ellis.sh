module load intel/compilers/17.1.132
module load cmake/3.5.2
module load boost/1.55.0/openmpi/1.10.4/intel/17.0.098
export OMP_NUM_THREADS=256
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
