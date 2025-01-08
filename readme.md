
# hermes_shmn

This library contains a variety of data structures and synchronization
primitives which are compatible with shared memory. This library is also
compatible with CUDA and ROCm.

[![Coverage Status](https://coveralls.io/repos/github/lukemartinlogan/hermes_shm/badge.svg?branch=master)](https://coveralls.io/github/lukemartinlogan/hermes_shm?branch=master)

## Installation

```
git clone https://github.com/lukemartinlogan/grc-repo.git
spack repo add grc-repo
spack install hermes_shm@dev
```

## Building

If you want to build the library yourself, do the following:
```
git clone https://github.com/lukemartinlogan/hermes_shm.git
cd hermes_shm
spack repo add scripts/hermes_shm
spack install hermes_shm
spack load --only dependencies hermes_shm

mkdir build
cd build
cmake ../ -DHERMES_ENABLE_CUDA=OFF -DHERMES_ENABLE_ROCM=OFF
make -j8
```

## CMake

### For CPU-Only Version
```
find_package(HermesShm CONFIG REQUIRED)
message(STATUS "found hermes_shm.h at ${HermesShm_INCLUDE_DIRS}")
target_link_libraries(HermesShm::cxx)
```

### For CUDA Version
```
find_package(HermesShm CONFIG REQUIRED)
message(STATUS "found hermes_shm.h at ${HermesShm_INCLUDE_DIRS}")
target_link_libraries(HermesShm::cudacxx)
```

### For ROCM Version
```
find_package(HermesShm CONFIG REQUIRED)
message(STATUS "found hermes_shm.h at ${HermesShm_INCLUDE_DIRS}")
target_link_libraries(HermesShm::rocmcxx)
```

## Tests

To run the tests, do the following:
```
ctest
```

To run the MPSC queue tests, do the following:
```
ctest -VV -R test_mpsc_queue_mpi
```