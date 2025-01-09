project(hermes_shm)



#------------------------------------------------------------------------------
# Test Cases
#------------------------------------------------------------------------------
set (LIBS
        hermes_shm_host
        Catch2::Catch2
        ${MPI_LIBS}
        ${OpenMP_LIBS})
add_executable(test_shm_exec
        ${TEST_MAIN}/main.cc
        test_init.cc
        test_shm.cc)
add_dependencies(test_shm_exec
        hermes_shm_host)
target_link_libraries(test_shm_exec ${LIBS})

add_test(NAME test_shm COMMAND ${CMAKE_BINARY_DIR}/bin/test_shm_exec)

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_shm_exec
        LIBRARY DESTINATION ${HERMES_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${HERMES_INSTALL_LIB_DIR}
        RUNTIME DESTINATION ${HERMES_INSTALL_BIN_DIR})

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HERMES_ENABLE_COVERAGE)
    set_coverage_flags(test_shm_exec)
endif()