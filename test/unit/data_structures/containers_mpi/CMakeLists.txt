cmake_minimum_required(VERSION 3.10)
project(hermes_shm)

set(CMAKE_CXX_STANDARD 17)

#------------------------------------------------------------------------------
# Test Cases
#------------------------------------------------------------------------------

add_executable(test_data_structure_mpi_exec
        ${TEST_MAIN}/main_mpi.cc
        test_init.cc
        mpsc_queue.cc
)

add_dependencies(test_data_structure_mpi_exec hermes_shm_data_structures)
target_link_libraries(test_data_structure_mpi_exec
        hermes_shm_data_structures Catch2::Catch2 ${MPI_LIBS} ${OpenMP_LIBS})

#------------------------------------------------------------------------------
# Build Tests
#------------------------------------------------------------------------------

# LIST TESTS
add_test(NAME test_mpsc_queue_mpi COMMAND
        mpirun -n 2 ${CMAKE_BINARY_DIR}/bin/test_data_structure_mpi_exec "TestMpscQueueMpi")

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_data_structure_mpi_exec
        EXPORT
        ${HERMES_EXPORTED_TARGETS}
        LIBRARY DESTINATION ${HERMES_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${HERMES_INSTALL_LIB_DIR}
        RUNTIME DESTINATION ${HERMES_INSTALL_BIN_DIR})

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HERMES_ENABLE_COVERAGE)
    set_coverage_flags(test_data_structure_mpi_exec)
endif()