#!/bin/bash -xe
rm -f CMakeCache.txt
rm -rf CMakeFiles

#
# Note on ROCM_SEARCH_PATH
#
# rocSOLVER and rocThurst are not installed on caraway 
# so one needs to do a local installation. 
# Then, ROCM_SEARCH_PATH should be a list of paths pointing to SOLVER, PRIM and Thrust
# for example for me is:
#"/ascldap/users/fnrizzi/repos/rocSOLVER/build/release/install;/ascldap/users/fnrizzi/repos/rocPRIM/install;/ascldap/users/fnrizzi/repos/rocThrust/install" \

cmake \
    -D CMAKE_CXX_COMPILER=hipcc \
    -D CMAKE_VERBOSE_MAKEFILE=On \
    -D CMAKE_CXX_STANDARD=14 \
    -D CMAKE_CXX_FLAGS="-g" \
    -D CMAKE_C_COMPILER=hipcc \
    -D CMAKE_C_FLAGS="-g" \
    -D Kokkos_ENABLE_HIP=ON \
    -D BUILD_SHARED_LIBS=ON \
    -D Kokkos_ARCH_VEGA908=ON \
    -D ROCM_SEARCH_PATH=<see_note_above>\
    ..
