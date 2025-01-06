cmake_minimum_required(VERSION 3.24)
set(GPU_RUNTIME "HIP" CACHE STRING "Switches between HIP and CUDA")
set(GPU_RUNTIMES "HIP" "CUDA")
set_property(CACHE GPU_RUNTIME PROPERTY STRINGS ${GPU_RUNTIMES})
project(hermes_shm_cuda LANGUAGES C CXX)

# Enable CUDA or ROCm
set(GPU_RUNTIME "HIP" CACHE STRING "Switches between HIP and CUDA")
if(NOT "${GPU_RUNTIME}" IN_LIST GPU_RUNTIMES)
    set(ERROR_MESSAGE
        "GPU_RUNTIME is set to \"${GPU_RUNTIME}\".\nGPU_RUNTIME must be either HIP or CUDA."
    )
    message(FATAL_ERROR ${ERROR_MESSAGE})
endif()
enable_language(${GPU_RUNTIME})
set(CMAKE_${GPU_RUNTIME}_STANDARD 17)
set(CMAKE_${GPU_RUNTIME}_EXTENSIONS OFF)
set(CMAKE_${GPU_RUNTIME}_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --forward-unknown-to-host-compiler")
set(ROCM_ROOT
        "/opt/rocm"
        CACHE PATH
        "Root directory of the ROCm installation"
)
if(GPU_RUNTIME STREQUAL "CUDA")
    list(APPEND include_dirs "${ROCM_ROOT}/include")
endif()

#------------------------------------------------------------------------------
# Build Tests
#------------------------------------------------------------------------------
add_executable(test_memory_rocm_exec
        backend_rocm.cc)
add_dependencies(test_memory_rocm_exec
        hermes_shm_data_structures)
target_link_libraries(test_memory_rocm_exec
        hermes_shm_data_structures)
set_source_files_properties(backend_rocm.cc PROPERTIES LANGUAGE ${GPU_RUNTIME})
target_compile_definitions(test_memory_rocm_exec PRIVATE HERMES_ENABLE_ROCM)

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_memory_rocm_exec
        EXPORT
        ${HERMES_EXPORTED_TARGETS}
        LIBRARY DESTINATION ${HERMES_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${HERMES_INSTALL_LIB_DIR}
        RUNTIME DESTINATION ${HERMES_INSTALL_BIN_DIR})

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HERMES_ENABLE_COVERAGE)
    set_coverage_flags(test_memory_rocm_exec)
endif()