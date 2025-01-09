project(hermes_shm_cuda)

#------------------------------------------------------------------------------
# Build Tests
#------------------------------------------------------------------------------
add_executable(test_memory_cuda_exec
        backend_cuda.cc)
add_dependencies(test_memory_cuda_exec
        hermes_shm_cuda)
target_link_libraries(test_memory_cuda_exec
        hermes_shm_cuda)
set_source_files_properties(backend_cuda.cc PROPERTIES LANGUAGE CUDA)
set_target_properties(test_memory_cuda_exec PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(test_memory_cuda_exec PRIVATE HERMES_ENABLE_CUDA)
target_compile_options(test_memory_cuda_exec PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_memory_cuda_exec
        LIBRARY DESTINATION ${HERMES_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${HERMES_INSTALL_LIB_DIR}
        RUNTIME DESTINATION ${HERMES_INSTALL_BIN_DIR})

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HERMES_ENABLE_COVERAGE)
    set_coverage_flags(test_memory_cuda_exec)
endif()