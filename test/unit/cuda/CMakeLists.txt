project(hermes_shm_cuda)

#------------------------------------------------------------------------------
# Build Tests
#------------------------------------------------------------------------------
add_cuda_executable(test_memory_cuda_exec FALSE backend_cuda.cc)
add_dependencies(test_memory_cuda_exec
        hermes_shm_cuda)
target_link_libraries(test_memory_cuda_exec
        hermes_shm_cuda)

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_memory_cuda_exec
        LIBRARY DESTINATION ${HSHM_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${HSHM_INSTALL_LIB_DIR}
        RUNTIME DESTINATION ${HSHM_INSTALL_BIN_DIR})

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HSHM_ENABLE_COVERAGE)
    set_coverage_flags(test_memory_cuda_exec)
endif()