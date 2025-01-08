message("BEGINNING TO FIND PACKAGE!!!")

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
    set(COMPRESS_INCLUDES 
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
    set(COMPRESS_LIB_DIRS
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
    set(ENCRYPT_INCLUDES ${libcrypto_INCLUDE_DIRS})
    set(ENCRYPT_LIB_DIRS ${libcrypto_LIBRARY_DIRS})
endif()

#-----------------------------------------------------------------------------
# Link to HSHM Dependencies
#-----------------------------------------------------------------------------
add_library(host_deps INTERFACE)
target_link_libraries(host_deps INTERFACE
        yaml-cpp
        ${REAL_TIME_FLAGS}
        ${SERIALIZATION_LIBS}
        ${COMPRESS_LIBS}
        ${ENCRYPT_LIBS}
)
target_link_directories(host_deps INTERFACE ${COMPRESS_LIB_DIRS} ${ENCRYPT_LIB_DIRS})
target_include_directories(host_deps INTERFACE ${COMPRESS_INCLUDES} ${ENCRYPT_INCLUDES})
if (HERMES_ENABLE_PTHREADS)
    target_link_libraries(host_deps INTERFACE pthread)
endif()

add_library(gpu_lib_deps INTERFACE)
if (HERMES_ENABLE_CUDA)
    target_compile_definitions(gpu_lib_deps INTERFACE HERMES_ENABLE_CUDA)
    target_compile_options(gpu_lib_deps INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()
if (HERMES_ENABLE_ROCM)
    target_compile_definitions(gpu_lib_deps INTERFACE HERMES_ENABLE_ROCM)
    target_compile_options(gpu_lib_deps INTERFACE -fgpu-rdc)
    target_link_libraries(gpu_lib_deps INTERFACE -fgpu-rdc)
    target_include_directories(gpu_lib_deps INTERFACE ${CMAKE_PREFIX_PATH}/hsa/include)
    target_link_libraries(gpu_lib_deps INTERFACE -fgpu-rdc)
endif()

add_library(gpu_exec_deps INTERFACE)
if (HERMES_ENABLE_CUDA)
    target_link_libraries(gpu_exec_deps INTERFACE gpu_lib_deps)
endif()
if (HERMES_ENABLE_ROCM)
    target_link_libraries(gpu_exec_deps INTERFACE gpu_exec_deps)
    target_link_libraries(gpu_exec_deps INTERFACE amdhip64 amd_comgr)
endif()

#-----------------------------------------------------------------------------
# Create HSHM Interfaces
#-----------------------------------------------------------------------------
add_library(cxx INTERFACE)
target_link_libraries(cxx INTERFACE
        hermes_shm_host
)
if (HERMES_ENABLE_CUDA)
    add_library(cudacxx INTERFACE)
    target_link_libraries(cudacxx INTERFACE
        hermes_shm_cuda
    )
endif()
if (HERMES_ENABLE_ROCM)
    add_library(rocmcxx INTERFACE)
    target_link_libraries(rocmcxx INTERFACE
        hermes_shm_rocm
    )
endif()
