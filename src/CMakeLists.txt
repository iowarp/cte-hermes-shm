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

set(CMAKE_CXX_STANDARD 17)

#-----------------------------------------------------------------------------
# Build HSHM
#-----------------------------------------------------------------------------
set (SOURCE_FILES
        memory_manager.cc
        data_structure_singleton.cc
)
if (BUILD_SHARED_LIBS)
    message("Building dynamic library")
    add_library(hermes_shm_data_structures SHARED ${SOURCE_FILES})
else()
    message("Building static library")
    add_library(hermes_shm_data_structures STATIC ${SOURCE_FILES})
endif()
target_link_libraries(hermes_shm_data_structures
        -lrt -ldl
        cereal::cereal
        $<$<BOOL:${HERMES_RPC_THALLIUM}>:thallium>
        ${COMPRESS_LIBS}
        ${ENCRYPT_LIBS}
)
if (HERMES_ENABLE_PTHREAD)
    target_link_libraries(hermes_shm_data_structures pthread)
endif()
if (HERMES_ENABLE_OPENMP)
    target_link_libraries(hermes_shm_data_structures OpenMP::OpenMP_CXX)
endif()
if (HERMES_ENABLE_CUDA)
    foreach (SOURCE IN LISTS SOURCE_FILES)
        set_source_files_properties(${SOURCE} PROPERTIES LANGUAGE CUDA)
        message(${SOURCE})
    endforeach()
    set_target_properties(hermes_shm_data_structures PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )
    if (BUILD_SHARED_LIBS)
        set_target_properties(hermes_shm_data_structures PROPERTIES
                POSITION_INDEPENDENT_CODE ON
        )
    endif()
    target_compile_definitions(hermes_shm_data_structures PRIVATE HERMES_ENABLE_CUDA)
    target_compile_options(hermes_shm_data_structures PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

#-----------------------------------------------------------------------------
# Add Target(s) to CMake Install
#-----------------------------------------------------------------------------
install(TARGETS
        hermes_shm_data_structures
        EXPORT
        ${HERMES_EXPORTED_TARGETS}
        LIBRARY DESTINATION ${HERMES_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${HERMES_INSTALL_LIB_DIR}
        RUNTIME DESTINATION ${HERMES_INSTALL_BIN_DIR})

#-----------------------------------------------------------------------------
# Export all exported targets to the build tree for use by parent project
#-----------------------------------------------------------------------------
set(HERMES_EXPORTED_LIBS
        hermes_shm_data_structures
        ${HERMES_EXPORTED_LIBS})
if(NOT HERMES_EXTERNALLY_CONFIGURED)
    EXPORT (
            TARGETS
            ${HERMES_EXPORTED_LIBS}
            FILE
            ${HERMES_EXPORTED_TARGETS}.cmake
    )
endif()

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HERMES_ENABLE_COVERAGE)
    set_coverage_flags(hermes_shm_data_structures)
endif()
