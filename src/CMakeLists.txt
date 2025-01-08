cmake_minimum_required(VERSION 3.10)
set(LANG C CXX)
if (HERMES_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES native)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --forward-unknown-to-host-compiler")
    set(LANG ${LANG} CUDA)
endif()
project(hermes_shm LANGUAGES ${LANG})



#-----------------------------------------------------------------------------
# Build HSHM
#-----------------------------------------------------------------------------
set(HSHM_LIBS "")
# BUILD HSHM FOR HOST ONLY
set (SOURCE_FILES
        memory_manager.cc
        system_info.cc)
add_library(hermes_shm_host ${SOURCE_FILES})
target_link_libraries(hermes_shm_host PUBLIC host_deps)
list(APPEND HSHM_LIBS hermes_shm_host)
# BUILD HSHM FOR CUDA ONLY
if (HERMES_ENABLE_CUDA)
    set(CUDA_SOURCE_FILES "")
    foreach (SOURCE IN LISTS SOURCE_FILES)
        set(CUDA_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/cuda/${SOURCE})
        configure_file(${SOURCE} ${CUDA_SOURCE} COPYONLY)
        set_source_files_properties(${CUDA_SOURCE} PROPERTIES LANGUAGE CUDA)
        list(APPEND CUDA_SOURCE_FILES ${CUDA_SOURCE}/${SOURCE})
    endforeach()
    add_library(hermes_shm_cuda ${CUDA_SOURCE_FILES})
    target_link_libraries(hermes_shm_cuda PUBLIC gpu_lib_deps)
    set_target_properties(hermes_shm_cuda PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )
    if (BUILD_SHARED_LIBS)
        set_target_properties(hermes_shm_host PROPERTIES
                POSITION_INDEPENDENT_CODE ON
        )
    endif()
    list(APPEND HSHM_LIBS hermes_shm_cuda)
endif()
# BUILD HSHM FOR ROCM ONLY
if (HERMES_ENABLE_ROCM)
    set(ROCM_SOURCE_FILES "")
    foreach (SOURCE IN LISTS SOURCE_FILES)
        set(ROCM_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/rocm/${SOURCE})
        configure_file(${SOURCE} ${ROCM_SOURCE} COPYONLY)
        set_source_files_properties(${ROCM_SOURCE} PROPERTIES LANGUAGE ${GPU_RUNTIME})
        list(APPEND ROCM_SOURCE_FILES ${ROCM_SOURCE})
    endforeach()
    add_library(hermes_shm_rocm ${ROCM_SOURCE_FILES})
    target_link_libraries(hermes_shm_rocm PUBLIC gpu_lib_deps)
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
        gpu_lib_deps
        host_deps
        EXPORT
        ${HERMES_EXPORTED_TARGETS}
        LIBRARY DESTINATION ${HERMES_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${HERMES_INSTALL_LIB_DIR}
        RUNTIME DESTINATION ${HERMES_INSTALL_BIN_DIR})

#-----------------------------------------------------------------------------
# Export all exported targets to the build tree for use by parent project
#-----------------------------------------------------------------------------
set(HERMES_EXPORTED_LIBS
        ${HSHM_LIBS}
        gpu_lib_deps
        host_deps
        ${HERMES_EXPORTED_LIBS})
EXPORT (
        TARGETS
        ${HERMES_EXPORTED_LIBS}
        FILE
        ${HERMES_EXPORTED_TARGETS}.cmake
)

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HERMES_ENABLE_COVERAGE)
    set_coverage_flags(hermes_shm_host)
endif()
