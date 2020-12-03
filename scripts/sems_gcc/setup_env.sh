module load sems-cmake/3.10.3
module load sems-gcc/7.3.0
module load sems-boost/1.69.0/base

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS=8
