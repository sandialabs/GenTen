% GT_INITIALIZE Initialize parallel resources for Genten
%
%  GT_INITIALIZE initializes parallel resources so Genten can run
%  parallel kernels using Kokkos.  Currently it does not accept any
%  options.
%
%  Driver functions such as GT_CP call this if it hasn't been called
%  by the user, so it usually doesn't need to be called directly.

%  See also GT_CP, GT_FINALIZE
