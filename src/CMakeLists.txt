project(hermes_shm)

#-----------------------------------------------------------------------------
# Build HSHM
#-----------------------------------------------------------------------------
set(HSHM_LIBS "")
# BUILD HSHM FOR HOST ONLY
set (SRC_FILES
        memory_manager.cc
        system_info.cc)
add_library(hermes_shm_host ${SRC_FILES})
target_link_libraries(hermes_shm_host PUBLIC host_deps)
list(APPEND HSHM_LIBS hermes_shm_host)
# BUILD HSHM FOR CUDA ONLY
if (HERMES_ENABLE_CUDA)
    add_cuda_library(hermes_shm_cuda TRUE ${SRC_FILES})
    list(APPEND HSHM_LIBS hermes_shm_cuda)
endif()
# BUILD HSHM FOR ROCM ONLY
if (HERMES_ENABLE_ROCM)
    add_rocm_library(hermes_shm_rocm TRUE ${SRC_FILES})
    target_link_libraries(hermes_shm_rocm PUBLIC gpu_rocm_lib_deps)
    list(APPEND HSHM_LIBS hermes_shm_rocm)
endif()

#-----------------------------------------------------------------------------
# Create HSHM Interfaces
#-----------------------------------------------------------------------------
add_library(cxx INTERFACE)
target_link_libraries(cxx INTERFACE
        hermes_shm_host
)
list(APPEND HSHM_LIBS cxx)
if (HERMES_ENABLE_CUDA)
    add_library(cudacxx INTERFACE)
    target_link_libraries(cudacxx INTERFACE
        hermes_shm_cuda
    )
    list(APPEND HSHM_LIBS cudacxx)
endif()
if (HERMES_ENABLE_ROCM)
    add_library(rocmcxx INTERFACE)
    target_link_libraries(rocmcxx INTERFACE
        hermes_shm_rocm
    )
    list(APPEND HSHM_LIBS rocmcxx)
endif()

#-----------------------------------------------------------------------------
# Add Target(s) to CMake Install
#-----------------------------------------------------------------------------
install(TARGETS
        ${HSHM_LIBS}
        gpu_rocm_lib_deps
        gpu_rocm_exec_deps
        gpu_cuda_lib_deps
        gpu_cuda_exec_deps
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
