cmake_minimum_required(VERSION 3.10)
project(hermes_shm)

set(CMAKE_CXX_STANDARD 17)

include_directories( ${Boost_INCLUDE_DIRS} )
include_directories( ${TEST_MAIN} )
add_executable(hshm_alloc_bench
        allocator.cc
)
add_dependencies(hshm_alloc_bench hermes_shm_data_structures)
target_link_libraries(hshm_alloc_bench
        hermes_shm_data_structures
        Catch2::Catch2
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        ${Boost_LIBRARIES})

#-----------------------------------------------------------------------------
# Add Target(s) to CMake Install
#-----------------------------------------------------------------------------
install(TARGETS
        hshm_alloc_bench
        EXPORT
        ${HERMES_EXPORTED_TARGETS}
        LIBRARY DESTINATION ${HERMES_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${HERMES_INSTALL_LIB_DIR}
        RUNTIME DESTINATION ${HERMES_INSTALL_BIN_DIR})
