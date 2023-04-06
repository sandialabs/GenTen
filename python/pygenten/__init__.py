from pygenten._pygenten import *
from pygenten.solvers import cp_als, cp_opt, gcp_opt, gcp_sgd
from pygenten.utils import read_and_distribute_tensor, distribute_tensor
try:
    from pygenten.helpers import make_ttb_tensor, make_ttb_ktensor
except:
    pass
import atexit

# Initialize GenTen when module is loaded (it internally checks if GenTen
# has already been initialized).  This initializes Kokkos and MPI (if enabled).
did_initialize = _pygenten.initializeGenten()

# Register function to finalize GenTen at exit, but only if we initialized it
if did_initialize:
    atexit.register(_pygenten.finalizeGenten)
