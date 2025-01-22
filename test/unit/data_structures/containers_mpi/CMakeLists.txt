project(hermes_shm)

if (HSHM_ENABLE_MPI)
    #------------------------------------------------------------------------------
    # Test Cases
    #------------------------------------------------------------------------------

    add_executable(test_data_structure_mpi_exec
            ${TEST_MAIN}/main_mpi.cc
            test_init.cc
            mpsc_queue.cc
    )

    add_dependencies(test_data_structure_mpi_exec hermes_shm_host)
    target_link_libraries(test_data_structure_mpi_exec
            hermes_shm_host Catch2::Catch2 ${MPI_LIBS} ${OpenMP_LIBS})
    message(STATUS "MPI_LIBS: ${MPI_LIBS}")

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
            LIBRARY DESTINATION ${HSHM_INSTALL_LIB_DIR}
            ARCHIVE DESTINATION ${HSHM_INSTALL_LIB_DIR}
            RUNTIME DESTINATION ${HSHM_INSTALL_BIN_DIR})

    #-----------------------------------------------------------------------------
    # Coverage
    #-----------------------------------------------------------------------------
    if(HSHM_ENABLE_COVERAGE)
        set_coverage_flags(test_data_structure_mpi_exec)
    endif()

endif()