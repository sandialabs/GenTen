## Example configure scripts

This directory contains example configure scripts for Genten and Kokkos for
various architectures.  In each directory, there is a script config-kokkos.sh
for configuring Kokkos for an external build, config-genten.sh for configuring
Genten for an external build of Kokkos, and config-genten-inline.sh for 
configuring Genten with an inline build of Kokkos.

There are also example scripts for loading modules that are needed for each
architecture, for various machines at Sandia Labs.  These are primarily only
useful for Sandia users, but can be useful to see what compilers Genten
is frequently compiled with.

The directory names should be self-explanatory, and follow a pattern of
architecture-host_architecture-compiler, where host_architecture is ommitted
for CPU and KNL architectures.