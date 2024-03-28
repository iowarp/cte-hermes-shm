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

set(BUILD_MPI_TESTS @BUILD_MPI_TESTS@)
set(BUILD_OpenMP_TESTS @BUILD_OpenMP_TESTS@)
set(BUILD_Boost_TESTS @BUILD_Boost_TESTS@)
set(HERMES_ENABLE_COMPRESS @HERMES_ENABLE_COMPRESS@)
set(HERMES_ENABLE_ENCRYPT @HERMES_ENABLE_ENCRYPT@)
set(HERMES_RPC_THALLIUM @HERMES_RPC_THALLIUM@)
set(HERMES_ENABLE_CEREAL @HERMES_ENABLE_CEREAL@)
set(HERMES_ENABLE_DOXYGEN @HERMES_ENABLE_DOXYGEN@)
set(HERMES_USE_ELF @HERMES_USE_ELF@)

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
        NAMES hermes_shm_data_structures
        HINTS ENV LD_LIBRARY_PATH ENV PATH
)
if( NOT HermesShm_LIBRARY )
  message(STATUS "FindHermesShm: Could not find hermes_shm_data_structures.so")
  set(HermesShm_FOUND OFF)
  return()
endif()

#-----------------------------------------------------------------------------
# Find all packages needed by hermes_shm
#-----------------------------------------------------------------------------
# Pkg-Config
find_package(PkgConfig REQUIRED)
if(PkgConfig)
  message(STATUS "found pkg config")
endif()

# RT
find_library(LIBRT rt)
if(NOT LIBRT)
  message(FATAL_ERROR "librt is required for POSIX shared memory")
endif()

# Libelf
if (HERMES_USE_ELF)
  pkg_check_modules(libelf REQUIRED libelf)
  message(STATUS "found libelf at ${libelf_INCLUDE_DIRS}")
  include_directories(${libelf_INCLUDE_DIRS})
  link_directories(${libelf_LIBRARY_DIRS})
endif()

# Cereal
if (HERMES_ENABLE_CEREAL)
  find_package(cereal CONFIG REQUIRED)
  message(STATUS "found cereal at ${cereal_DIR}")
endif()

# Catch2
#find_package(Catch2 3.0.1 REQUIRED)
find_package(Catch2 REQUIRED)
message(STATUS "found catch2.h at ${Catch2_CXX_INCLUDE_DIRS}")

# YAML-CPP
find_package(yaml-cpp REQUIRED)
message(STATUS "found yaml-cpp at ${yaml-cpp_DIR}")

# MPI
if(BUILD_MPI_TESTS)
  find_package(MPI REQUIRED COMPONENTS C CXX)
  set(MPI_LIBS MPI::MPI_CXX)
  message(STATUS "found mpi.h at ${MPI_CXX_INCLUDE_DIRS}")
endif()

# OpenMP
if(BUILD_OpenMP_TESTS)
  find_package(OpenMP REQUIRED COMPONENTS C CXX)
  set(OpenMP_LIBS OpenMP::OpenMP_CXX)
  message(STATUS "found omp.h at ${OpenMP_CXX_INCLUDE_DIRS}")
endif()

# thallium
if(HERMES_RPC_THALLIUM)
  find_package(thallium CONFIG REQUIRED)
  if(thallium_FOUND)
    message(STATUS "found thallium at ${thallium_DIR}")
  endif()
endif()

# Boost
if(BUILD_Boost_TESTS)
  find_package(Boost REQUIRED COMPONENTS regex system filesystem fiber REQUIRED)
  message(STATUS "found boost at ${Boost_INCLUDE_DIRS}")
endif()

if(HERMES_ENABLE_COMPRESS)
  pkg_check_modules(bzip2 REQUIRED bzip2)
  message(STATUS "found bz2.h at ${bzip2_INCLUDE_DIRS}")

  pkg_check_modules(lzo2 REQUIRED lzo2)
  message(STATUS "found lzo2.h at ${lzo2_INCLUDE_DIRS}")
  get_filename_component(lzo2_dir "${lzo2_INCLUDE_DIRS}" DIRECTORY)

  pkg_check_modules(libzstd REQUIRED libzstd)
  message(STATUS "found zstd.h at ${libzstd_INCLUDE_DIRS}")

  pkg_check_modules(liblz4 REQUIRED liblz4)
  message(STATUS "found lz4.h at ${liblz4_INCLUDE_DIRS}")

  pkg_check_modules(zlib REQUIRED zlib)
  message(STATUS "found zlib.h at ${zlib_INCLUDE_DIRS}")

  pkg_check_modules(liblzma REQUIRED liblzma)
  message(STATUS "found liblzma.h at ${liblzma_INCLUDE_DIRS}")

  pkg_check_modules(libbrotlicommon REQUIRED libbrotlicommon libbrotlidec libbrotlienc)
  message(STATUS "found libbrotli.h at ${libbrotlicommon_INCLUDE_DIRS}")

  pkg_check_modules(snappy REQUIRED snappy)
  message(STATUS "found libbrotli.h at ${snappy_INCLUDE_DIRS}")

  pkg_check_modules(blosc2 REQUIRED blosc2)
  message(STATUS "found blosc2.h at ${blosc2_INCLUDE_DIRS}")

  set(COMPRESS_LIBRARIES
          bz2
          ${lzo2_LIBRARIES}
          ${libzstd_LIBRARIES}
          ${liblz4_LIBRARIES}
          ${zlib_LIBRARIES}
          ${liblzma_LIBRARIES}
          ${libbrotlicommon_LIBRARIES}
          ${snappy_LIBRARIES}
          ${blosc2_LIBRARIES}
  )
  set(COMPRESS_INCLUDE_DIRS
          ${bzip2_INCLUDE_DIRS}
          ${lzo2_INCLUDE_DIRS} ${lzo2_dir}
          ${libzstd_INCLUDE_DIRS}
          ${liblz4_INCLUDE_DIRS}
          ${zlib_INCLUDE_DIRS}
          ${liblzma_INCLUDE_DIRS}
          ${libbrotlicommon_INCLUDE_DIRS}
          ${snappy_INCLUDE_DIRS}
          ${blosc2_INCLUDE_DIRS}
  )
  set(COMPRESS_LIBRARY_DIRS
          ${bzip2_LIBRARY_DIRS}
          ${lzo2_LIBRARY_DIRS}
          ${libzstd_LIBRARY_DIRS}
          ${liblz4_LIBRARY_DIRS}
          ${zlib_LIBRARY_DIRS}
          ${liblzma_LIBRARY_DIRS}
          ${libbrotlicommon_LIBRARY_DIRS}
          ${snappy_LIBRARY_DIRS}
          ${blosc2_LIBRARY_DIRS}
  )
endif()

if(HERMES_ENABLE_ENCRYPT)
  pkg_check_modules(libcrypto REQUIRED libcrypto)
  message(STATUS "found libcrypto.h at ${libcrypto_INCLUDE_DIRS}")

  set(ENCRYPT_LIBRARIES ${libcrypto_LIBRARIES})
  set(ENCRYPT_INCLUDE_DIRS ${libcrypto_INCLUDE_DIRS})
  set(ENCRYPT_LIBRARY_DIRS ${libcrypto_LIBRARY_DIRS})
endif()

#-----------------------------------------------------------------------------
# Mark hermes as found and set all needed packages
#-----------------------------------------------------------------------------
set(HermesShm_LIBRARY_DIR "")
get_filename_component(HermesShm_LIBRARY_DIRS ${HermesShm_LIBRARY} PATH)
# Set uncached variables as per standard.
set(HermesShm_FOUND ON)
set(HermesShm_INCLUDE_DIRS
        ${ENCRYPT_INCLUDE_DIRS}
        ${COMPRESS_INCLUDE_DIRS}
        ${HermesShm_INCLUDE_DIR})
set(HermesShm_LIBRARIES
        -lrt -ldl cereal::cereal -lstdc++fs
        ${ENCRYPT_LIBRARIES}
        ${COMPRESS_LIBRARIES}
        ${HermesShm_LIBRARY}
        ${MPI_LIBS}
        ${OpenMP_LIBS})
set(HermesShm_LIBRARY_DIRS
        ${ENCRYPT_LIBRARY_DIRS}
        ${COMPRESS_LIBRARY_DIRS})
