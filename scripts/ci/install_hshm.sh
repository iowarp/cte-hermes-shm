#!/bin/bash

# CD into git workspace
cd ${GITHUB_WORKSPACE}

# Set spack env
set +x
SPACK_DIR=${INSTALL_DIR}/spack
. ${SPACK_DIR}/share/spack/setup-env.sh
set -x

set -x
set -e
set -o pipefail

mkdir -p "${HOME}/install"
mkdir build
cd build
# export CXXFLAGS="${CXXFLAGS} -std=c++17 -Werror -Wall -Wextra"
source ${HOME}/.bashrc
spack load --only dependencies hermes_shm
cmake ../ -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$[HOME}/install
cmake --build . -- -j4
ctest -VV