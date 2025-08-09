
# Hermes Shared Memory (CTE)

[![IoWarp](https://img.shields.io/badge/IoWarp-GitHub-blue.svg)](http://github.com/iowarp)
[![GRC](https://img.shields.io/badge/GRC-Website-blue.svg)](https://grc.iit.edu/)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Compatible-green.svg)](https://developer.nvidia.com/cuda-zone)
[![ROCm](https://img.shields.io/badge/ROCm-Compatible-red.svg)](https://rocmdocs.amd.com/)

A high-performance shared memory library containing data structures and synchronization primitives compatible with shared memory, CUDA, and ROCm.

## Dependencies

### System Requirements
- CMake >= 3.10
- C++17 compatible compiler
- Optional: CUDA toolkit (for GPU support)
- Optional: ROCm (for AMD GPU support)
- Optional: MPI, ZeroMQ, Thallium (for distributed computing)

## Installation

### Spack Installation (Users)

For those installing this component (rather than all of iowarp):
```bash
git clone https://github.com/grc-iit/grc-repo.git
spack repo add grc-repo
spack install cte-hermes-shm
```

### Manual Installation (Developers)

#### Install Dependencies

This will install dependencies of cte-hermes-shm:
```bash
git clone https://github.com/grc-iit/grc-repo.git
spack repo add grc-repo
spack install cte-hermes-shm +nocompile
spack load cte-hermes-shm +nocompile
```

NOTE: spack load needs to be done for each new terminal.

#### Build from Source

This will compile:
```bash
git clone https://github.com/grc-iit/cte-hermes-shm.git
cd cte-hermes-shm
mkdir build
cd build
cmake ../ -DHSHM_ENABLE_CUDA=OFF -DHSHM_ENABLE_ROCM=OFF
make -j8
```

## CMake

### For CPU-Only Version
```
find_package(HermesShm CONFIG REQUIRED)
message(STATUS "found cte-hermes-shm.h at ${HermesShm_INCLUDE_DIRS}")
target_link_libraries(hshm::cxx)
```

### For CUDA Version
```
find_package(HermesShm CONFIG REQUIRED)
message(STATUS "found cte-hermes-shm.h at ${HermesShm_INCLUDE_DIRS}")
target_link_libraries(hshm::cudacxx)
```

### For ROCM Version
```
find_package(HermesShm CONFIG REQUIRED)
message(STATUS "found cte-hermes-shm.h at ${HermesShm_INCLUDE_DIRS}")
target_link_libraries(hshm::rocmcxx_gpu)
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

## Project Structure

- `include/cte-hermes-shm/` - Core library headers
  - `data_structures/` - Shared memory data structures
  - `memory/` - Memory management and allocators
  - `thread/` - Threading and synchronization primitives
  - `util/` - Utility functions and helpers
- `src/` - Core library implementation
- `test/` - Unit tests and benchmarks
- `benchmark/` - Performance benchmarks
- `CMake/` - CMake configuration files
