#!/bin/bash

set -ex

if test $# -ne 1; then
    echo "Usage: ./install-dpc++.sh <sycl-version>"
    exit 1
fi

sycl_version=${1}
sycl_url=https://github.com/intel/llvm/archive/sycl-nightly
sycl_archive="${sycl_version}".tar.gz
extraction_root_dir=$(pwd)/dpc++
extraction_dir="${extraction_root_dir}"/llvm-sycl-nightly-"${sycl_version}"
sycl_install_dir=/opt/sycl

mkdir -p "${extraction_root_dir}"
cd "${extraction_root_dir}"
wget --quiet "${sycl_url}"/"${sycl_archive}"
tar -xf "${sycl_archive}"

cd "${extraction_dir}"
python buildbot/configure.py --cuda
python buildbot/compile.py

sudo rm -rf "${sycl_install_dir}"
sudo mkdir -p "${sycl_install_dir}"
sudo cp -r "${extraction_dir}"/build/install/* "${sycl_install_dir}"
sudo ldconfig
