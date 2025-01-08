cmake_minimum_required(VERSION 3.24)
project(hermes_shm_cuda LANGUAGES C CXX)

#------------------------------------------------------------------------------
# Build Tests
#------------------------------------------------------------------------------
add_executable(test_memory_rocm_exec
        backend_rocm.cc)
add_dependencies(test_memory_rocm_exec
        hermes_shm_rocm)
target_link_libraries(test_memory_rocm_exec hermes_shm_rocm)
set_source_files_properties(backend_rocm.cc PROPERTIES LANGUAGE ${GPU_RUNTIME})

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_memory_rocm_exec
        LIBRARY DESTINATION ${HERMES_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${HERMES_INSTALL_LIB_DIR}
        RUNTIME DESTINATION ${HERMES_INSTALL_BIN_DIR})

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HERMES_ENABLE_COVERAGE)
    set_coverage_flags(test_memory_rocm_exec)
endif()