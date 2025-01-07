cmake_minimum_required(VERSION 3.24)
project(hermes_shm_cuda LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

#------------------------------------------------------------------------------
# Build Tests
#------------------------------------------------------------------------------
# add_executable(test_memory_rocm_exec
#         backend_rocm.cc)
# add_dependencies(test_memory_rocm_exec
#         hermes_shm_data_structures)
# target_link_libraries(test_memory_rocm_exec
#         hermes_shm_data_structures)
# set_source_files_properties(backend_rocm.cc PROPERTIES LANGUAGE ${GPU_RUNTIME})
# target_compile_definitions(test_memory_rocm_exec PRIVATE HERMES_ENABLE_ROCM) 

# Compile test_lib.cc into a library
add_library(test_lib SHARED test_lib.cc)
set_source_files_properties(test_lib.cc PROPERTIES LANGUAGE ${GPU_RUNTIME})
target_compile_definitions(test_lib PRIVATE HERMES_ENABLE_ROCM)

# Add test_lib_main
add_executable(test_lib_main test_lib_main.cc)
target_link_libraries(test_lib_main test_lib)
set_source_files_properties(test_lib_main.cc PROPERTIES LANGUAGE ${GPU_RUNTIME})
target_compile_definitions(test_lib_main PRIVATE HERMES_ENABLE_ROCM)

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