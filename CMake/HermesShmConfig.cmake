# Find HermesShm header and library.
#

# This module defines the following uncached variables:
#  HermesShm_FOUND, if false, do not try to use hermes_shm.
#  HermesShm_LIBRARY_DIRS, the directory where the hermes_shm library is found.

#-----------------------------------------------------------------------------
# Define constants
#-----------------------------------------------------------------------------
set(HSHM_SHM_VERSION_MAJOR @HSHM_SHM_VERSION_MAJOR@)
set(HSHM_SHM_VERSION_MINOR @HSHM_SHM_VERSION_MINOR@)
set(HSHM_SHM_VERSION_PATCH @HSHM_SHM_VERSION_PATCH@)

set(HSHM_ENABLE_MPI @HSHM_ENABLE_MPI@)
set(HSHM_RPC_THALLIUM @HSHM_RPC_THALLIUM@)
set(HSHM_ENABLE_OPENMP @HSHM_ENABLE_OPENMP@)
set(HSHM_ENABLE_CEREAL @HSHM_ENABLE_CEREAL@)
set(HSHM_ENABLE_COVERAGE @HSHM_ENABLE_COVERAGE@)
set(HSHM_ENABLE_DOXYGEN @HSHM_ENABLE_DOXYGEN@)
set(HSHM_ENABLE_WINDOWS_THREADS @HSHM_ENABLE_WINDOWS_THREADS@)
set(HSHM_ENABLE_PTHREADS @HSHM_ENABLE_PTHREADS@)
set(HSHM_DEBUG_LOCK @HSHM_DEBUG_LOCK@)
set(HSHM_ENABLE_COMPRESS @HSHM_ENABLE_COMPRESS@)
set(HSHM_ENABLE_ENCRYPT @HSHM_ENABLE_ENCRYPT@)
set(HSHM_USE_ELF @HSHM_USE_ELF@)
set(HSHM_ENABLE_CUDA @HSHM_ENABLE_CUDA@)
set(HSHM_ENABLE_ROCM @HSHM_ENABLE_ROCM@)
set(HSHM_NO_COMPILE @HSHM_NO_COMPILE@)

set(REAL_TIME_FLAGS @REAL_TIME_FLAGS@)

# Find the HermesShm Package
find_package(HermesShmCore REQUIRED)

# Find the HermesShm dependencies
find_package(HermesShmCommon REQUIRED)

# Add my library to RPATH
list(APPEND CMAKE_INSTALL_RPATH "@HSHM_INSTALL_LIB_DIR@")

# Basic: Target link directories / includes
target_include_directories(hshm::cxx INTERFACE "@HSHM_INSTALL_INCLUDE_DIR@")
target_link_directories(hshm::cxx INTERFACE "@HSHM_INSTALL_LIB_DIR@")

# CUDA: Target link directories / includes
if (HSHM_ENABLE_CUDA)
    target_include_directories(hshm::cudacxx INTERFACE "@HSHM_INSTALL_INCLUDE_DIR@")
    target_link_directories(hshm::cudacxx INTERFACE "@HSHM_INSTALL_LIB_DIR@")
endif()

# ROCm: Target link directories / includes
if (HSHM_ENABLE_ROCM)
    execute_process(COMMAND hipconfig --rocmpath
        OUTPUT_VARIABLE rocm_path)
    message(STATUS "ROCm SDK path: ${rocm_path}")

    # TODO(llogan): This is a hack to make vscode detect HIP headers and not show errors
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -isystem ${rocm_path}/include -D__HIP_PLATFORM_AMD__")
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -isystem @HSHM_INSTALL_INCLUDE_DIR@")

    # GPU code
    target_include_directories(hshm::rocmcxx_gpu INTERFACE "@HSHM_INSTALL_INCLUDE_DIR@")
    target_link_directories(hshm::rocmcxx_gpu INTERFACE "@HSHM_INSTALL_LIB_DIR@")

    # Host-only code
    target_include_directories(hshm::rocmcxx_host INTERFACE "@HSHM_INSTALL_INCLUDE_DIR@")
    target_link_directories(hshm::rocmcxx_host INTERFACE "@HSHM_INSTALL_LIB_DIR@")
endif()
