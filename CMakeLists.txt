cmake_minimum_required(VERSION 3.25)
project(hermes_shm LANGUAGES C CXX)
include(CTest)
set(CMAKE_CXX_STANDARD 17)

# ------------------------------------------------------------------------------
# Global variables
# ------------------------------------------------------------------------------
set(HSHM_SHM_VERSION_MAJOR 2)
set(HSHM_SHM_VERSION_MINOR 0)
set(HSHM_SHM_VERSION_PATCH 0)
set(HSHM_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${HSHM_ROOT}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
add_compile_definitions(_CRT_SECURE_NO_DEPRECATE)

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(BUILD_SHARED_LIBS "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)" ON)
option(HSHM_BUILD_TESTS "Build tests" ON)
option(HSHM_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(HSHM_ENABLE_CMAKE_DOTENV "Build benchmarks" OFF)

option(HSHM_ENABLE_MPI "Enable MPI support" OFF)
option(HSHM_RPC_THALLIUM "Build tests which depend on thallium" OFF)
option(HSHM_ENABLE_OPENMP "Enable the use of OpenMP" OFF)
option(HSHM_ENABLE_CEREAL "Enable serialization using cereal" ON)
option(HSHM_ENABLE_COVERAGE "Check how well tests cover code" OFF)
option(HSHM_ENABLE_DOXYGEN "Check how well the code is documented" OFF)

option(HSHM_ENABLE_WINDOWS_THREADS "Support spawning windows threads" OFF)
option(HSHM_ENABLE_PTHREADS "Support spawning pthreads" OFF)
option(HSHM_DEBUG_LOCK "Used for debugging locks" OFF)
option(HSHM_ENABLE_COMPRESS "Enable compression" OFF)
option(HSHM_ENABLE_ENCRYPT "Enable encryption" OFF)
option(HSHM_ENABLE_ELF "Enable elf" OFF)
option(HSHM_ENABLE_CUDA "Enable CUDA support" OFF)
option(HSHM_ENABLE_ROCM "Enable ROCm support" OFF)
option(HSHM_NO_COMPILE "Disable compiling / installing this library" OFF)

if(WIN32)
    message(STATUS "Detected Windows OS")
    set(HSHM_ENABLE_WINDOWS_SYSINFO ON)
    set(HSHM_ENABLE_WINDOWS_THREADS ON)
else()
    message(STATUS "Detected UNIX OS")
    set(HSHM_ENABLE_PROCFS_SYSINFO ON)
    set(HSHM_ENABLE_PTHREADS ON)
endif()

if(HSHM_NO_COMPILE)
    install(
        FILES
        ${HSHM_ROOT}/include/hermes_shm/constants/settings.h_templ
        DESTINATION
        ${CMAKE_INSTALL_PREFIX}/include/hermes_shm/constants/
    )
    return()
endif()

# ------------------------------------------------------------------------------
# DOTENV
# ------------------------------------------------------------------------------
if(HSHM_ENABLE_CMAKE_DOTENV)
    if(EXISTS ${CMAKE_SOURCE_DIR}/.env.cmake)
        include(.env.cmake)
    endif()
endif()

# ------------------------------------------------------------------------------
# Setup CMake Environment
# ------------------------------------------------------------------------------
set(HSHM_EXPORTED_TARGETS "HermesShm")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Executables.")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Libraries")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all static libraries.")

# ------------------------------------------------------------------------------
# Setup install and output Directories
# ------------------------------------------------------------------------------
if(NOT HSHM_INSTALL_BIN_DIR)
    set(HSHM_INSTALL_BIN_DIR ${CMAKE_INSTALL_PREFIX}/bin)
endif()

if(NOT HSHM_INSTALL_LIB_DIR)
    set(HSHM_INSTALL_LIB_DIR ${CMAKE_INSTALL_PREFIX}/lib)
endif()

if(NOT HSHM_INSTALL_INCLUDE_DIR)
    set(HSHM_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include)
endif()

if(NOT HSHM_INSTALL_DATA_DIR)
    set(HSHM_INSTALL_DATA_DIR ${CMAKE_INSTALL_PREFIX}/share)
endif()

# ------------------------------------------------------------------------------
# CMake Modules
# ------------------------------------------------------------------------------
if(NOT IS_HSHM_MAIN)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/CMake)
    message(${CMAKE_MODULE_PATH})
endif()

# ------------------------------------------------------------------------------
# Optimization
# ------------------------------------------------------------------------------

# set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$" OR CMAKE_CXX_COMPILER_ID STREQUAL "MinGW" OR CMAKE_CXX_COMPILER_ID MATCHES "^Intel")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("IN DEBUG MODE")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
        set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -g -O0")
        add_compile_definitions(HSHM_DEBUG)
    else()
        message("IN RELEASE MODE")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")
    endif()

    set(HSHM_COMPILER_GNU "ON")

    if(APPLE)
        set(REAL_TIME_FLAGS "-ldl")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "OpenBSD")
        set(REAL_TIME_FLAGS "")
    else()
        set(REAL_TIME_FLAGS "-lrt -ldl")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
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

# ------------------------------------------------------------------------------
# External libraries
# ------------------------------------------------------------------------------
include(CMake/HermesShmCommonConfig.cmake)

# This is for compatability with CLANG + Spack
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_INSTALL_RPATH "${HSHM_INSTALL_LIB_DIR}")
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# ENABLE GPU SUPPORT
if(HSHM_ENABLE_CUDA)
    hshm_enable_cuda(17)
endif()

if(HSHM_ENABLE_ROCM)
    hshm_enable_rocm("HIP" 17)
endif()

# HOST DEPENDENCIES
add_library(host_deps INTERFACE)

if(NOT YAML_CPP_LIBRARIES)
    set(YAML_CPP_LIBRARIES yaml-cpp)
endif()

target_link_libraries(host_deps INTERFACE
    ${YAML_CPP_LIBRARIES}
    ${REAL_TIME_FLAGS}
    ${SERIALIZATION_LIBS}
    ${COMPRESS_LIBS}
    ${ENCRYPT_LIBS}
    ${Boost_LIBRARIES}
    ${ELF_LIBS}
)
target_link_directories(host_deps INTERFACE
    ${COMPRESS_LIB_DIRS} ${ENCRYPT_LIB_DIRS} ${Boost_LIBRARY_DIRS}
    ${ELF_LIB_DIRS} ${YAML_CPP_LIBRARY_DIR})
target_include_directories(host_deps INTERFACE
    ${COMPRESS_INCLUDES} ${ENCRYPT_INCLUDES} ${Boost_INCLUDE_DIRS}
    ${ELF_INCLUDES} ${YAML_CPP_INCLUDE_DIR})

if(HSHM_ENABLE_PTHREADS)
    target_link_libraries(host_deps INTERFACE pthread)
endif()

# -----------------------------------------------------------------------------
# Documentation
# -----------------------------------------------------------------------------
if(HSHM_ENABLE_DOXYGEN)
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

# ------------------------------------------------------------------------------
# Code Coverage
# ------------------------------------------------------------------------------
if(NOT IS_HSHM_MAIN)
    if(HSHM_ENABLE_COVERAGE)
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

# ------------------------------------------------------------------------------
# Build hermes_shm
# ------------------------------------------------------------------------------
add_subdirectory(src)

if(NOT IS_HSHM_MAIN)
    add_custom_target(lint COMMAND bash ${HSHM_ROOT}/scripts/lint.sh ${HSHM_ROOT})
    add_custom_target(preamble COMMAND python3 ${HSHM_ROOT}/scripts/preamble.py ${HSHM_ROOT})
endif()

# ------------------------------------------------------------------------------
# Build tests + benchmarks
# ------------------------------------------------------------------------------
set(TEST_MAIN ${HSHM_ROOT}/test/unit)
enable_testing()

if(HSHM_BUILD_TESTS)
    message("Building HSHM unit tests")
    add_subdirectory(test)
endif()

if(HSHM_BUILD_BENCHMARKS)
    message("Building HSHM benchmarks")
    add_subdirectory(benchmark)
endif()

# add_subdirectory(example)

# ------------------------------------------------------------------------------
# Install hshm
# ------------------------------------------------------------------------------
install(DIRECTORY include DESTINATION ${CMAKE_INSTALL_PREFIX})

# Configure the hermes_shm options
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/hermes_shm/constants/settings.h_templ
    ${CMAKE_CURRENT_SOURCE_DIR}/include/hermes_shm/constants/settings.h @ONLY
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/CMake/HermesShmConfig.cmake
    ${PROJECT_BINARY_DIR}/CMakeFiles/HermesShmConfig.cmake @ONLY
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/CMake/HermesShmCommonConfig.cmake
    ${PROJECT_BINARY_DIR}/CMakeFiles/HermesShmCommonConfig.cmake @ONLY
)

install(
    FILES
    ${PROJECT_BINARY_DIR}/CMakeFiles/HermesShmConfig.cmake
    ${PROJECT_BINARY_DIR}/CMakeFiles/HermesShmCommonConfig.cmake
    DESTINATION
    ${CMAKE_INSTALL_PREFIX}/cmake
)

install(EXPORT ${HSHM_EXPORTED_TARGETS}
    FILE ${HSHM_EXPORTED_TARGETS}CoreConfig.cmake
    NAMESPACE hshm::
    DESTINATION cmake
)

jarvis_repo_add("${HSHM_ROOT}/test/jarvis_hshm" "")
