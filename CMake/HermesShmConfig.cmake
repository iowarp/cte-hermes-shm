# Find labstor header and library.
#

# This module defines the following uncached variables:
#  HermesShm_FOUND, if false, do not try to use hermes_shm.
#  HermesShm_INCLUDE_DIRS, where to find hermes_shm.h.
#  HermesShm_LIBRARIES, the libraries to link against to use the hermes_shm library
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

#-----------------------------------------------------------------------------
# Find hermes_shm header
#-----------------------------------------------------------------------------
find_path(
  HermesShm_INCLUDE_DIR
        hermes_shm/hermes_shm.h
  HINTS ENV PATH ENV CPATH
)
if( NOT HermesShm_INCLUDE_DIR )
  message(STATUS "FindHermesShm: Could not find hermes_shm.h")
  set(HermesShm_FOUND OFF)
  return()
endif()
get_filename_component(HermesShm_DIR ${HermesShm_INCLUDE_DIR} PATH)

#-----------------------------------------------------------------------------
# Find hermes_shm library
#-----------------------------------------------------------------------------
find_library(
        HermesShm_LIBRARY
        NAMES hermes_shm_host
        HINTS ENV LD_LIBRARY_PATH ENV PATH
)
if( NOT HermesShm_LIBRARY )
  message(STATUS "FindHermesShm: Could not find hermes_shm_host.so")
  set(HermesShm_FOUND OFF)
  return()
endif()

#-----------------------------------------------------------------------------
# Find all packages needed by hermes_shm
#-----------------------------------------------------------------------------
include(HermesShmCommon.cmake)

#-----------------------------------------------------------------------------
# Mark hermes as found and set all needed packages
#-----------------------------------------------------------------------------
set(HermesShm_LIBRARY_DIR "")
get_filename_component(HermesShm_LIBRARY_DIRS ${HermesShm_LIBRARY} PATH)
# Set uncached variables as per standard.
set(HermesShm_FOUND ON)
