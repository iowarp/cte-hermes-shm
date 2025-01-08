#------------------------------------------------------------------------------
# External libraries
#------------------------------------------------------------------------------
# This is for compatability with SPACK
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Pkg-Config
find_package(PkgConfig REQUIRED)
if(PkgConfig)
  message(STATUS "found pkg config")
endif()

# Catch2
find_package(Catch2 3.0.1 REQUIRED)
message(STATUS "found catch2.h at ${Catch2_CXX_INCLUDE_DIRS}")

# YAML-CPP
find_package(yaml-cpp REQUIRED)
message(STATUS "found yaml-cpp at ${yaml-cpp_DIR}")

# MPICH
if(HERMES_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS C CXX)
    set(MPI_LIBS MPI::MPI_CXX)
    message(STATUS "found mpi.h at ${MPI_CXX_INCLUDE_DIRS}")
endif()

# ROCm
if (HERMES_ENABLE_ROCM)
    find_package(HIP REQUIRED)
endif()

# OpenMP
if(HERMES_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS C CXX)
    set(OPENMP_LIBS OpenMP::OpenMP_CXX)
    message(STATUS "found omp.h at ${OpenMP_CXX_INCLUDE_DIRS}")
endif()

# thallium
if(HERMES_RPC_THALLIUM)
    find_package(thallium CONFIG REQUIRED)
    if(thallium_FOUND)
        message(STATUS "found thallium at ${thallium_DIR}")
    endif()
    set(SERIALIZATION_LIBS thallium ${SERIALIZATION_LIBS})
endif()

# Cereal
if(HERMES_ENABLE_CEREAL)
    find_package(cereal CONFIG REQUIRED)
    if(cereal_FOUND)
        message(STATUS "found cereal at ${cereal_DIR}")
    endif()
    set(SERIALIZATION_LIBS cereal::cereal ${SERIALIZATION_LIBS})
endif()

# Boost
find_package(Boost REQUIRED)
message(STATUS "found boost.h at ${Boost_INCLUDE_DIRS}")

# Compression libraries
if(HERMES_ENABLE_COMPRESS)
    message("HERMES_ENABLE_COMPRESS is ON")
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

    set(COMPRESS_LIBS
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
    include_directories(
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
    link_directories(
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

# Encryption libraries
if(HERMES_ENABLE_ENCRYPT)
    pkg_check_modules(libcrypto REQUIRED libcrypto)
    message(STATUS "found libcrypto.h at ${libcrypto_INCLUDE_DIRS}")

    set(ENCRYPT_LIBS ${libcrypto_LIBRARIES})
    include_directories(${libcrypto_INCLUDE_DIRS})
    link_directories(${libcrypto_LIBRARY_DIRS})
endif()

#-----------------------------------------------------------------------------
# Link to HSHM Dependencies
#-----------------------------------------------------------------------------
add_library(hermes_shm_deps INTERFACE)
target_link_libraries(hermes_shm_deps INTERFACE
        yaml-cpp
        ${REAL_TIME_FLAGS}
        ${SERIALIZATION_LIBS}
        ${COMPRESS_LIBS}
        ${ENCRYPT_LIBS}
)
if (HERMES_ENABLE_PTHREADS)
    target_link_libraries(hermes_shm_deps INTERFACE pthread)
endif()
if (HERMES_ENABLE_CUDA)
    target_compile_definitions(hermes_shm_deps INTERFACE HERMES_ENABLE_CUDA)
    target_compile_options(hermes_shm_deps INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()
if (HERMES_ENABLE_ROCM)
    target_compile_definitions(hermes_shm_deps INTERFACE HERMES_ENABLE_ROCM)
    target_compile_options(hermes_shm_deps INTERFACE -fgpu-rdc)
    target_link_libraries(hermes_shm_deps INTERFACE -fgpu-rdc)
    target_include_directories(hermes_shm_deps INTERFACE ${CMAKE_PREFIX_PATH}/hsa/include)
endif()

#-----------------------------------------------------------------------------
# Link to HSHM
#-----------------------------------------------------------------------------
add_library(hermes_shm INTERFACE)
target_link_libraries(hermes_shm_deps INTERFACE
        hermes_shm_data_structures
        hermes_shm_deps
)
add_dependencies(hermes_shm hermes_shm)