cmake_minimum_required(VERSION 3.25)
project(hermes_shm LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

#------------------------------------------------------------------------------
# Global variables
#------------------------------------------------------------------------------
set(HERMES_SHM_VERSION_MAJOR 2)
set(HERMES_SHM_VERSION_MINOR 0)
set(HERMES_SHM_VERSION_PATCH 0)
set(HERMES_SHM_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${HERMES_SHM_ROOT}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
add_compile_definitions(_CRT_SECURE_NO_DEPRECATE)

#------------------------------------------------------------------------------
# Options
#------------------------------------------------------------------------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(BUILD_SHARED_LIBS "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)" ON)
option(BUILD_HSHM_TESTS "Build tests" ON)
option(BUILD_HSHM_BENCHMARKS "Build benchmarks" OFF)

option(HERMES_ENABLE_MPI "Enable MPI support" OFF)
option(HERMES_RPC_THALLIUM "Build tests which depend on thallium" OFF)
option(HERMES_ENABLE_OPENMP "Enable the use of OpenMP" OFF)
option(HERMES_ENABLE_CEREAL "Enable serialization using cereal" ON)
option(HERMES_ENABLE_COVERAGE "Check how well tests cover code" OFF)
option(HERMES_ENABLE_DOXYGEN "Check how well the code is documented" OFF)

option(HERMES_ENABLE_WINDOWS_THREADS "Support spawning windows threads" OFF)
option(HERMES_ENABLE_PTHREADS "Support spawning pthreads" OFF)
option(HERMES_DEBUG_LOCK "Used for debugging locks" OFF)
option(HERMES_ENABLE_COMPRESS "Enable compression" OFF)
option(HERMES_ENABLE_ENCRYPT "Enable encryption" OFF)
option(HERMES_USE_ELF "Enable elf" OFF)
option(HERMES_ENABLE_CUDA "Enable CUDA support" OFF)
option(HERMES_ENABLE_ROCM "Enable ROCm support" OFF)
option(HERMES_NO_COMPILE "Disable compiling / installing this library" OFF)

if (WIN32)
    message(STATUS "Detected Windows OS")
    set(HERMES_ENABLE_WINDOWS_SYSINFO ON)
    set(HERMES_ENABLE_WINDOWS_THREADS ON)
else()
    message(STATUS "Detected non-Windows OS")
    set(HERMES_ENABLE_PROCFS_SYSINFO ON)
    set(HERMES_ENABLE_PTHREADS ON)
endif()

if (HERMES_NO_COMPILE)
    install(
            FILES
            ${CMAKE_SOURCE_DIR}/include/hermes_shm/constants/settings.h_templ
            DESTINATION
            ${CMAKE_INSTALL_PREFIX}/include/hermes_shm/constants/
    )
    return()
endif()

#------------------------------------------------------------------------------
# Setup CMake Environment
#------------------------------------------------------------------------------
set(HERMES_EXPORTED_TARGETS "HermesShm")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Executables.")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Libraries")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all static libraries.")

#------------------------------------------------------------------------------
# Setup install and output Directories
#------------------------------------------------------------------------------
if(NOT HERMES_INSTALL_BIN_DIR)
    set(HERMES_INSTALL_BIN_DIR ${CMAKE_INSTALL_PREFIX}/bin)
endif()
if(NOT HERMES_INSTALL_LIB_DIR)
    set(HERMES_INSTALL_LIB_DIR ${CMAKE_INSTALL_PREFIX}/lib)
endif()
if(NOT HERMES_INSTALL_INCLUDE_DIR)
    set(HERMES_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include)
endif()
if(NOT HERMES_INSTALL_DATA_DIR)
    set(HERMES_INSTALL_DATA_DIR ${CMAKE_INSTALL_PREFIX}/share)
endif()

#------------------------------------------------------------------------------
# CMake Modules
#------------------------------------------------------------------------------
if (NOT IS_HERMES_MAIN)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/CMake)
    message(${CMAKE_MODULE_PATH})
endif()

#------------------------------------------------------------------------------
# Optimization
#------------------------------------------------------------------------------

# set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "MinGW" OR CMAKE_CXX_COMPILER STREQUAL "Intel")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("IN DEBUG MODE")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
        add_compile_definitions(HSHM_DEBUG)
    else()
        message("IN RELEASE MODE")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")
    endif()
    set(HSHM_COMPILER_GNU "ON")
    set(REAL_TIME_FLAGS "-lrt -ldl")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("IN DEBUG MODE")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi /Od")
        add_compile_definitions(HSHM_DEBUG)
    else()
        message("IN RELEASE MODE")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
    endif()
    set(HSHM_COMPILER_MSVC "ON")
else()
    message(FATAL_ERROR "Unsupported compiler ${CMAKE_CXX_COMPILER_ID}")
endif()


#------------------------------------------------------------------------------
# External libraries
#------------------------------------------------------------------------------
include(CMake/HermesShmCommonConfig.cmake) 

# This is for compatability with CLANG + Spack
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_INSTALL_RPATH "${HERMES_INSTALL_LIB_DIR}")
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# ENABLE GPU SUPPORT
if (HERMES_ENABLE_CUDA)
    hermes_enable_cuda(17)
endif()
if (HERMES_ENABLE_ROCM)
    hermes_enable_rocm("HIP" 17)
endif()

# HOST DEPENDENCIES
add_library(host_deps INTERFACE)
target_link_libraries(host_deps INTERFACE
        yaml-cpp
        ${REAL_TIME_FLAGS}
        ${SERIALIZATION_LIBS}
        ${COMPRESS_LIBS}
        ${ENCRYPT_LIBS}
        ${Boost_LIBRARIES}
)
target_link_directories(host_deps INTERFACE 
    ${COMPRESS_LIB_DIRS} ${ENCRYPT_LIB_DIRS} ${Boost_LIBRARY_DIRS})
target_include_directories(host_deps INTERFACE 
    ${COMPRESS_INCLUDES} ${ENCRYPT_INCLUDES} ${Boost_INCLUDE_DIRS})
if (HERMES_ENABLE_PTHREADS)
    target_link_libraries(host_deps INTERFACE pthread)
endif()

# CUDA (GPU) LIB DEPS
add_library(cuda_gpu_lib_deps INTERFACE)
add_library(HermesShm::cuda_gpu_lib_deps ALIAS cuda_gpu_lib_deps)
if (HERMES_ENABLE_CUDA)
    target_compile_definitions(cuda_gpu_lib_deps INTERFACE HERMES_ENABLE_CUDA)
    target_compile_options(cuda_gpu_lib_deps INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

# CUDA (GPU) EXEC DEPS
add_library(cuda_gpu_exec_deps INTERFACE)
add_library(HermesShm::cuda_gpu_exec_deps ALIAS cuda_gpu_exec_deps)
if (HERMES_ENABLE_CUDA)
    target_link_libraries(cuda_gpu_exec_deps INTERFACE cuda_gpu_lib_deps) 
endif()

# ROCM (GPU) LIB DEPS
add_library(rocm_gpu_lib_deps INTERFACE)
add_library(HermesShm::rocm_gpu_lib_deps ALIAS rocm_gpu_lib_deps)
if (HERMES_ENABLE_ROCM)
    target_link_libraries(rocm_gpu_lib_deps INTERFACE -fgpu-rdc host_deps)
    target_compile_options(rocm_gpu_lib_deps INTERFACE -fgpu-rdc)
    target_compile_definitions(rocm_gpu_lib_deps INTERFACE HERMES_ENABLE_ROCM)
    # target_include_directories(rocm_gpu_lib_deps INTERFACE ${CMAKE_PREFIX_PATH}/hsa/include)
endif()

# ROCM (HOST) LIB DEPS
add_library(rocm_host_lib_deps INTERFACE)
add_library(HermesShm::rocm_host_lib_deps ALIAS rocm_host_lib_deps)
if (HERMES_ENABLE_ROCM)
    target_link_libraries(rocm_host_lib_deps INTERFACE host_deps)
endif()

# ROCM (HOST) EXEC DEPS
add_library(rocm_host_exec_deps INTERFACE)
add_library(HermesShm::rocm_host_exec_deps ALIAS rocm_host_exec_deps)
if (HERMES_ENABLE_ROCM)
    # target_link_libraries(rocm_host_exec_deps INTERFACE rocm_host_lib_deps amdhip64 amd_comgr)
    target_link_libraries(rocm_host_exec_deps INTERFACE rocm_host_lib_deps)
endif()

# ROCM (GPU) EXEC DEPS
add_library(rocm_gpu_exec_deps INTERFACE)
add_library(HermesShm::rocm_gpu_exec_deps ALIAS rocm_gpu_exec_deps)
if (HERMES_ENABLE_ROCM)
    target_link_libraries(rocm_gpu_exec_deps INTERFACE rocm_gpu_lib_deps amdhip64 amd_comgr)
endif()

#-----------------------------------------------------------------------------
# Documentation
#-----------------------------------------------------------------------------
if(HERMES_ENABLE_DOXYGEN)
    include(UseDoxygenDoc)

    add_doxygen_doc(
            BUILD_DIR
            ${CMAKE_CURRENT_BINARY_DIR}/_build
            DOXY_FILE
            ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in
            TARGET_NAME
            dox
            COMMENT
            "HTML documentation"
    )
endif()

#------------------------------------------------------------------------------
# Code Coverage
#------------------------------------------------------------------------------
if (NOT IS_HERMES_MAIN)
    if(HERMES_ENABLE_COVERAGE)
        set(COVERAGE_FLAGS "-fprofile-arcs -ftest-coverage" CACHE STRING
            "Flags to the coverage program to perform coverage inspection")
        mark_as_advanced(COVERAGE_FLAGS)

        macro(set_coverage_flags target)
            set_target_properties(${target}
                    PROPERTIES
                    COMPILE_FLAGS ${COVERAGE_FLAGS}
                    LINK_FLAGS ${COVERAGE_FLAGS}
                    )
        endmacro()
    endif()
endif()

#------------------------------------------------------------------------------
# Build hermes_shm
#------------------------------------------------------------------------------
add_subdirectory(src)

if (NOT IS_HERMES_MAIN)
    add_custom_target(lint COMMAND bash ${HERMES_SHM_ROOT}/scripts/lint.sh ${HERMES_SHM_ROOT})
    add_custom_target(preamble COMMAND python3 ${HERMES_SHM_ROOT}/scripts/preamble.py ${HERMES_SHM_ROOT})
endif()

#------------------------------------------------------------------------------
# Build tests + benchmarks
#------------------------------------------------------------------------------
set(TEST_MAIN ${HERMES_SHM_ROOT}/test/unit)
enable_testing()
if (BUILD_HSHM_TESTS)
  message("Building HSHM unit tests")
  add_subdirectory(test)
endif()
if (BUILD_HSHM_BENCHMARKS)
  message("Building HSHM benchmarks")
  add_subdirectory(benchmark)
endif()
#add_subdirectory(example)

#------------------------------------------------------------------------------
# Install hshm
#------------------------------------------------------------------------------
install(DIRECTORY include DESTINATION ${CMAKE_INSTALL_PREFIX})

# Configure the hermes_shm options
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/include/hermes_shm/constants/settings.h_templ
        ${PROJECT_BINARY_DIR}/include/hermes_shm/constants/settings.h @ONLY
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/CMake/HermesShmConfig.cmake
        ${PROJECT_BINARY_DIR}/CMakeFiles/HermesShmConfig.cmake @ONLY
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/CMake/HermesShmCommonConfig.cmake
        ${PROJECT_BINARY_DIR}/CMakeFiles/HermesShmCommonConfig.cmake @ONLY
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/CMake/UseDoxygenDoc.cmake
        ${PROJECT_BINARY_DIR}/CMakeFiles/UseDoxygenDoc.cmake @ONLY
)

install(
    FILES
        ${PROJECT_BINARY_DIR}/CMakeFiles/HermesShmConfig.cmake
        ${PROJECT_BINARY_DIR}/CMakeFiles/HermesShmCommonConfig.cmake
        ${PROJECT_BINARY_DIR}/CMakeFiles/UseDoxygenDoc.cmake
    DESTINATION
        ${CMAKE_INSTALL_PREFIX}/cmake
)

install(
        FILES
        ${PROJECT_BINARY_DIR}/include/hermes_shm/constants/settings.h
        DESTINATION
        ${CMAKE_INSTALL_PREFIX}/include/hermes_shm/constants/
)

install(EXPORT ${HERMES_EXPORTED_TARGETS}
  FILE ${HERMES_EXPORTED_TARGETS}CoreConfig.cmake
  NAMESPACE HermesShm::
  DESTINATION cmake
)
