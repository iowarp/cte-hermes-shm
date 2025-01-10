# Find HermesShm header and library.
#

# This module defines the following uncached variables:
#  HermesShm_FOUND, if false, do not try to use hermes_shm.
#  HermesShm_LIBRARY_DIRS, the directory where the hermes_shm library is found.

#-----------------------------------------------------------------------------
# Define constants
#-----------------------------------------------------------------------------
set(HERMES_SHM_VERSION_MAJOR @HERMES_SHM_VERSION_MAJOR@)
set(HERMES_SHM_VERSION_MINOR @HERMES_SHM_VERSION_MINOR@)
set(HERMES_SHM_VERSION_PATCH @HERMES_SHM_VERSION_PATCH@)

set(HERMES_ENABLE_MPI @HERMES_ENABLE_MPI@)
set(HERMES_RPC_THALLIUM @HERMES_RPC_THALLIUM@)
set(HERMES_ENABLE_OPENMP @HERMES_ENABLE_OPENMP@)
set(HERMES_ENABLE_CEREAL @HERMES_ENABLE_CEREAL@)
set(HERMES_ENABLE_COVERAGE @HERMES_ENABLE_COVERAGE@)
set(HERMES_ENABLE_DOXYGEN @HERMES_ENABLE_DOXYGEN@)
set(HERMES_ENABLE_WINDOWS_THREADS @HERMES_ENABLE_WINDOWS_THREADS@)
set(HERMES_ENABLE_PTHREADS @HERMES_ENABLE_PTHREADS@)
set(HERMES_DEBUG_LOCK @HERMES_DEBUG_LOCK@)
set(HERMES_ENABLE_COMPRESS @HERMES_ENABLE_COMPRESS@)
set(HERMES_ENABLE_ENCRYPT @HERMES_ENABLE_ENCRYPT@)
set(HERMES_USE_ELF @HERMES_USE_ELF@)
set(HERMES_ENABLE_CUDA @HERMES_ENABLE_CUDA@)
set(HERMES_ENABLE_ROCM @HERMES_ENABLE_ROCM@)
set(HERMES_NO_COMPILE @HERMES_NO_COMPILE@)

set(REAL_TIME_FLAGS @REAL_TIME_FLAGS@)

# Find the HermesShm Package
find_package(HermesShmCore REQUIRED)

# Find the HermesShm dependencies
find_package(HermesShmCommon REQUIRED)

target_include_directories(HermesShm::cxx INTERFACE "@CMAKE_INSTALL_PREFIX@/include")
if (HERMES_ENABLE_CUDA)
    target_include_directories(HermesShm::cudacxx INTERFACE "@CMAKE_INSTALL_PREFIX@/include")
endif()
if (HERMES_ENABLE_ROCM)
    target_include_directories(HermesShm::rocmcxx INTERFACE "@CMAKE_INSTALL_PREFIX@/include")
endif()
