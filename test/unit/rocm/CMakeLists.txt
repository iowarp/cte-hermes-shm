project(hermes_shm_cuda LANGUAGES C CXX)

#------------------------------------------------------------------------------
# Build Tests
#------------------------------------------------------------------------------
add_rocm_gpu_executable(test_memory_rocm_exec FALSE "backend_rocm.cc")
add_dependencies(test_memory_rocm_exec
        hermes_shm_rocm_gpu)
target_link_libraries(test_memory_rocm_exec PUBLIC hermes_shm_rocm_gpu)

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_memory_rocm_exec
        LIBRARY DESTINATION ${HSHM_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${HSHM_INSTALL_LIB_DIR}
        RUNTIME DESTINATION ${HSHM_INSTALL_BIN_DIR})

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HSHM_ENABLE_COVERAGE)
    set_coverage_flags(test_memory_rocm_exec)
endif()