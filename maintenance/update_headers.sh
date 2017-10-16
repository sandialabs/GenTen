#!/bin/bash

# This script relies on a refactoring script in Trilinos to update all of the
# copyright headers throughout the repo.

# Run this from the top-level genten directory.

SCRIPT=${HOME}/Trilinos/Trilinos/commonTools/refactoring/update-copyright-header.py
HEADER_DIR=$PWD/maintenance

find . -name \*.hpp -not -path ./kokkos/\* -exec $SCRIPT --copyright-header=${HEADER_DIR}/copyright_header_cpp.txt --file={} \;
find . -name \*.cpp -not -path ./kokkos/\* -exec $SCRIPT --copyright-header=${HEADER_DIR}/copyright_header_cpp.txt --file={} \;
find . -name \*.c -not -path ./kokkos/\* -exec $SCRIPT --copyright-header=${HEADER_DIR}/copyright_header_cpp.txt --file={} \;
find . -name CMakeLists.txt -not -path ./kokkos/\* -exec $SCRIPT --copyright-header=${HEADER_DIR}/copyright_header_shell.txt --file={} \;
find . -name \*.cmake -not -path ./kokkos/\* -not -name CMakeInclude.h.cmake -exec $SCRIPT --copyright-header=${HEADER_DIR}/copyright_header_shell.txt --file={} \;
find . -name \*.py -not -path ./kokkos/\* -exec $SCRIPT --copyright-header=${HEADER_DIR}/copyright_header_shell.txt --file={} \;
find . -name CMakeInclude.h.cmake -not -path ./kokkos/\* -exec $SCRIPT --copyright-header=${HEADER_DIR}/copyright_header_cpp.txt --file={} \;
