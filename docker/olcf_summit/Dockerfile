# Generates a Docker container with an installed GenTen built with gcc
# compilers and CUDA parallelism for execution on summit.
#
# This is intended to be run from the top-level genten source tree

# Base image
FROM code.ornl.gov:4567/olcfcontainers/olcfbaseimages/mpiimage-centos-cuda:latest
MAINTAINER Eric Phipps <etphipp@sandia.gov>

# Change to bash so we have, e.g., -o pipefail
SHELL ["/bin/bash", "-c"]

# Install GenTen dependencies
#RUN dnf -y install epel-release && dnf -y install fakeroot
#RUN fakeroot dnf upgrade -y && fakeroot dnf update -y
RUN fakeroot dnf install -y cmake vim

# Add genten src
RUN mkdir -p /src
COPY . /src/genten

# Setup environment
ENV \
  GENTEN_SRC_PATH=/src/genten \
  GENTEN_INSTALL_PATH=/usr \
  OMPI_CXX=/src/genten/tpls/kokkos/bin/nvcc_wrapper \
  CUDA_MANAGED_FORCE_DEVICE_ALLOC=1

# Build and install genten
RUN \
  set -o pipefail && \
  mkdir -p /build/genten  && \
  cd /build/genten && \
  /src/genten/docker/olcf_summit/do_configure.sh 2>&1 | tee cmake_output.txt && \
  make -j8 install 2>&1 | tee make_output.txt
