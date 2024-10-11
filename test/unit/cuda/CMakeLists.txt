cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "60;70;80")
project(hermes_shm_cuda LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

#------------------------------------------------------------------------------
# Build Tests
#------------------------------------------------------------------------------
add_executable(test_memory_cuda_exec
        backend_cuda.cc)
set_source_files_properties(backend_cuda.cc PROPERTIES LANGUAGE CUDA)
set_target_properties(test_memory_cuda_exec PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(test_memory_cuda_exec PRIVATE HERMES_ENABLE_CUDA)

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_memory_cuda_exec
        EXPORT
        ${HERMES_EXPORTED_TARGETS}
        LIBRARY DESTINATION ${HERMES_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${HERMES_INSTALL_LIB_DIR}
        RUNTIME DESTINATION ${HERMES_INSTALL_BIN_DIR})

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HERMES_ENABLE_COVERAGE)
    set_coverage_flags(test_memory_cuda_exec)
endif()