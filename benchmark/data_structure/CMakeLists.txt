cmake_minimum_required(VERSION 3.10)
project(hermes_shm)

set(CMAKE_CXX_STANDARD 17)

include_directories( ${Boost_INCLUDE_DIRS} )
include_directories( ${TEST_MAIN} )
add_executable(benchmark_data_structures
        ${TEST_MAIN}/main.cc
        test_init.cc
        atomic.cc
        ref.cc
        string.cc
        list.cc
        vector.cc
        unordered_map.cc
        queue.cc
        lock.cc
)
add_dependencies(benchmark_data_structures hermes_shm_data_structures)
target_link_libraries(benchmark_data_structures
        hermes_shm_data_structures
        Catch2::Catch2
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        ${Boost_LIBRARIES})

#-----------------------------------------------------------------------------
# Add Target(s) to CMake Install
#-----------------------------------------------------------------------------
install(TARGETS
        benchmark_data_structures
        EXPORT
        ${HERMES_EXPORTED_TARGETS}
        LIBRARY DESTINATION ${HERMES_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${HERMES_INSTALL_LIB_DIR}
        RUNTIME DESTINATION ${HERMES_INSTALL_BIN_DIR})
