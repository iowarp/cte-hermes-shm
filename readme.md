
# hermes_shm

This library contains a variety of data structures and synchronization
primitives which are compatible with shared memory.

[![Coverage Status](https://coveralls.io/repos/github/lukemartinlogan/hermes_shm/badge.svg?branch=master)](https://coveralls.io/github/lukemartinlogan/hermes_shm?branch=master)

## Installation

```
git clone https://github.com/lukemartinlogan/hermes_shm.git
cd hermes_shm
spack repo add scripts/hermes_shm
spack install hermes_shm
spack load hermes_shm
```

## Building

If you want to build the library yourself, do the following:
```
git clone https://github.com/lukemartinlogan/hermes_shm.git
cd hermes_shm
spack repo add scripts/hermes_shm
spack install hermes_shm
spack load --only=dependencies hermes_shm

mkdir build
cd build
cmake ../
make -j8
```

## CMake

To include hermes_shm in your cmake project, do:
```
find_package(HermesShm CONFIG REQUIRED)
message(STATUS "found hermes_shm.h at ${HermesShm_INCLUDE_DIRS}")
include_directories(${HermesShm_INCLUDE_DIRS})
target_link_libraries(my_target ${HermesShm_LIBRARIES})
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
