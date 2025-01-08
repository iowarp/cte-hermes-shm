cmake_minimum_required(VERSION 3.24)
project(hermes_shm_cuda LANGUAGES C CXX)

#------------------------------------------------------------------------------
# Build Tests
#------------------------------------------------------------------------------
add_executable(test_memory_rocm_exec
        backend_rocm.cc)
add_dependencies(test_memory_rocm_exec
        hermes_shm_data_structures)
set_source_files_properties(backend_rocm.cc PROPERTIES LANGUAGE ${GPU_RUNTIME})
target_compile_definitions(test_memory_rocm_exec PRIVATE HERMES_ENABLE_ROCM) 
target_compile_options(test_memory_rocm_exec PRIVATE -fgpu-rdc)
set_source_files_properties(${TEST_SOURCES} PROPERTIES LANGUAGE ${GPU_RUNTIME})
if (WIN32)
  target_link_libraries(test_memory_rocm_exec PRIVATE -lhermes_shm_data_structures -L${CMAKE_CURRENT_BINARY_DIR})
else()
  target_link_libraries(test_memory_rocm_exec PRIVATE hermes_shm_data_structures)
endif()
target_link_libraries(test_memory_rocm_exec PRIVATE -fgpu-rdc amdhip64 amd_comgr)

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
# install(TARGETS
#         test_memory_rocm_exec
#         EXPORT
#         ${HERMES_EXPORTED_TARGETS}
#         LIBRARY DESTINATION ${HERMES_INSTALL_LIB_DIR}
#         ARCHIVE DESTINATION ${HERMES_INSTALL_LIB_DIR}
#         RUNTIME DESTINATION ${HERMES_INSTALL_BIN_DIR})

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HERMES_ENABLE_COVERAGE)
    set_coverage_flags(test_memory_rocm_exec)
endif()