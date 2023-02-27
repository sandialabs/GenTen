from pygenten._pygenten import *
from pygenten.solvers import cp_als, cp_opt, gcp_opt, gcp_sgd
import atexit

# Initialize GenTen when module is loaded (it internally checks if GenTen
# has already been initialized).  This initializes Kokkos and MPI (if enabled).
_pygenten.initializeGenten()

# Register function to finalize GenTen at exit
atexit.register(_pygenten.finalizeGenten)
