# Find labstor header and library.
#

# This module defines the following uncached variables:
#  HermesShm_FOUND, if false, do not try to use labstor.
#  HermesShm_INCLUDE_DIRS, where to find labstor.h.
#  HermesShm_LIBRARIES, the libraries to link against to use the labstor library
#  HermesShm_LIBRARY_DIRS, the directory where the labstor library is found.

find_path(
  HermesShm_INCLUDE_DIR
        hermes_shm/hermes_shm.h
  HINTS ENV PATH ENV CPATH
)

set(HERMES_SHM_VERSION_MAJOR @HERMES_SHM_VERSION_MAJOR@)
set(HERMES_SHM_VERSION_MINOR @HERMES_SHM_VERSION_MINOR@)

set(HERMES_ENABLE_COMPRESS @HERMES_ENABLE_COMPRESS@)
set(HERMES_ENABLE_ENCRYPT @HERMES_ENABLE_ENCRYPT@)

if( HermesShm_INCLUDE_DIR )
  message(STATUS "FindHermesShm: Could not find hermes_shm.h")
  set(HermesShm_FOUND OFF)
  return()
endif()

get_filename_component(HermesShm_DIR ${HermesShm_INCLUDE_DIR} PATH)

#-----------------------------------------------------------------------------
# Find all packages needed by hermes_shm
#-----------------------------------------------------------------------------
find_library(
  HermesShm_LIBRARY
  NAMES hermes_shm_data_structures
  HINTS ENV LD_LIBRARY_PATH ENV PATH
)
# RT
find_library(LIBRT rt)
if(NOT LIBRT)
  message(FATAL_ERROR "librt is required for POSIX shared memory")
endif()

# Cereal
find_package(cereal CONFIG REQUIRED)
if(cereal_FOUND)
  message(STATUS "found cereal at ${cereal_DIR}")
endif()

if(HERMES_ENABLE_COMPRESS)
  find_package(BZip2 REQUIRED)
  message(STATUS "found bz2.h at ${BZip2_INCLUDE_DIRS}")

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
          ${BZip2_INCLUDE_DIRS}
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
          ${BZip2_LIBRARY_DIRS}
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
if( HermesShm_LIBRARY )
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
          ${HermesShm_LIBRARY})
  set(HermesShm_LIBRARY_DIRS
          ${ENCRYPT_LIBRARY_DIRS}
          ${COMPRESS_LIBRARY_DIRS})
endif(HermesShm_LIBRARY)

if(HermesShm_FOUND)
  if(NOT HermesShm_FIND_QUIETLY)
    message(STATUS "FindHermesShm: Found both hermes_shm.h and libhermes_shm.a")
  endif(NOT HermesShm_FIND_QUIETLY)
else(HermesShm_FOUND)
  if(HermesShm_FIND_REQUIRED)
    message(STATUS "FindHermesShm: Could not find hermes_shm.h and/or libhermes_shm.a")
  endif(HermesShm_FIND_REQUIRED)
endif(HermesShm_FOUND)
