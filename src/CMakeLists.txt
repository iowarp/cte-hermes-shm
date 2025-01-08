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
# BUILD HSHM FOR HOST-ONLY
set (SOURCE_FILES
        memory_manager.cc
        system_info.cc
        data_structure_singleton.cc)
add_library(hermes_shm_host ${SOURCE_FILES})
target_link_libraries(hermes_shm_host PUBLIC host_deps)

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
endif()
if (HERMES_ENABLE_ROCM)
    set(ROCM_SOURCE_FILES "")
    foreach (SOURCE IN LISTS SOURCE_FILES)
        set(ROCM_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/rocm/${SOURCE})
        configure_file(${SOURCE} ${ROCM_SOURCE} COPYONLY)
        set_source_files_properties(${ROCM_SOURCE} PROPERTIES LANGUAGE ${GPU_RUNTIME})
        list(APPEND ROCM_SOURCE_FILES ${ROCM_SOURCE})
    endforeach()
    message(STATUS "ROCM_SOURCE_FILES: ${ROCM_SOURCE_FILES}")
    add_library(hermes_shm_rocm ${ROCM_SOURCE_FILES})
    target_link_libraries(hermes_shm_rocm PUBLIC gpu_lib_deps)
endif()

#-----------------------------------------------------------------------------
# Add Target(s) to CMake Install
#-----------------------------------------------------------------------------
install(TARGETS
        hermes_shm_host
        EXPORT
        ${HERMES_EXPORTED_TARGETS}
        LIBRARY DESTINATION ${HERMES_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${HERMES_INSTALL_LIB_DIR}
        RUNTIME DESTINATION ${HERMES_INSTALL_BIN_DIR})

#-----------------------------------------------------------------------------
# Export all exported targets to the build tree for use by parent project
#-----------------------------------------------------------------------------
# set(HERMES_EXPORTED_LIBS
#         hermes_shm_host
#         ${HERMES_EXPORTED_LIBS})
# if(NOT HERMES_EXTERNALLY_CONFIGURED)
#     EXPORT (
#             TARGETS
#             ${HERMES_EXPORTED_LIBS}
#             FILE
#             ${HERMES_EXPORTED_TARGETS}.cmake
#     )
# endif()

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HERMES_ENABLE_COVERAGE)
    set_coverage_flags(hermes_shm_host)
endif()
