#!/bin/bash

# set -x
# set -e
# set -o pipefail

mkdir build
pushd build

INSTALL_PREFIX="${HOME}/${LOCAL}"

export CXXFLAGS="${CXXFLAGS} -std=c++17 -Werror -Wall -Wextra"
cmake                                                      \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}               \
    -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}                  \
    -DCMAKE_BUILD_RPATH=${INSTALL_PREFIX}/lib              \
    -DCMAKE_INSTALL_RPATH=${INSTALL_PREFIX}/lib            \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE}                       \
    -DBUILD_SHARED_LIBS=ON                                 \
    -DHERMES_ENABLE_DOXYGEN=ON                             \
    -DHERMES_ENABLE_COVERAGE=ON                            \
    ..
make dox >& out.txt
# cat out.txt | grep warning | grep -v "ignoring unsupported tag"
popd
rec="$( grep warning build/out.txt | grep -v "ignoring unsupported tag" |  wc -l )"
echo "$rec"
exit "$rec"

