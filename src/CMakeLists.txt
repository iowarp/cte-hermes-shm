project(hermes_shm)

#-----------------------------------------------------------------------------
# Build HSHM
#-----------------------------------------------------------------------------
set(HSHM_LIBS "")

# BUILD HSHM FOR HOST ONLY
set (SRC_FILES
    # system_info.cc   # NOTE(llogan): Commented out because ROCm is annoying and buggy
    memory_manager.cc)
add_library(hermes_shm_host ${SRC_FILES})
target_link_libraries(hermes_shm_host PUBLIC host_deps)
list(APPEND HSHM_LIBS hermes_shm_host)
add_library(cxx INTERFACE)
target_link_libraries(cxx INTERFACE hermes_shm_host)
list(APPEND HSHM_LIBS cxx)

# BUILD HSHM FOR CUDA ONLY
if (HERMES_ENABLE_CUDA)
    add_cuda_library(hermes_shm_cuda TRUE ${SRC_FILES})
    target_link_libraries(hermes_shm_cuda PUBLIC cuda_gpu_lib_deps)
    list(APPEND HSHM_LIBS hermes_shm_cuda)
    add_library(cudacxx INTERFACE)
    target_link_libraries(cudacxx INTERFACE hermes_shm_cuda)
    list(APPEND HSHM_LIBS cudacxx)
endif()

# BUILD HSHM FOR ROCM ONLY
if (HERMES_ENABLE_ROCM)
    add_rocm_library(hermes_shm_rocm_gpu TRUE ${SRC_FILES})
    target_link_libraries(hermes_shm_rocm_gpu PUBLIC rocm_gpu_lib_deps)
    list(APPEND HSHM_LIBS hermes_shm_rocm_gpu)
    add_library(rocmcxx_gpu INTERFACE)
    target_link_libraries(rocmcxx_gpu INTERFACE hermes_shm_rocm_gpu)
    list(APPEND HSHM_LIBS rocmcxx_gpu)

    add_rocm_host_library(hermes_shm_rocm_host TRUE ${SRC_FILES})
    target_link_libraries(hermes_shm_rocm_host PUBLIC rocm_host_lib_deps)
    list(APPEND HSHM_LIBS hermes_shm_rocm_host)
    add_library(rocmcxx_host INTERFACE)
    target_link_libraries(rocmcxx_host INTERFACE hermes_shm_rocm_host)
    list(APPEND HSHM_LIBS rocmcxx_host)
endif()

#-----------------------------------------------------------------------------
# Add Target(s) to CMake Install
#-----------------------------------------------------------------------------
install(TARGETS
        ${HSHM_LIBS}
        rocm_gpu_lib_deps
        rocm_host_lib_deps
        rocm_gpu_exec_deps
        cuda_gpu_lib_deps
        cuda_gpu_exec_deps
        host_deps
        EXPORT
        ${HERMES_EXPORTED_TARGETS}
        LIBRARY DESTINATION ${HERMES_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${HERMES_INSTALL_LIB_DIR}
        RUNTIME DESTINATION ${HERMES_INSTALL_BIN_DIR})

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HERMES_ENABLE_COVERAGE)
    set_coverage_flags(hermes_shm_host)
endif()
