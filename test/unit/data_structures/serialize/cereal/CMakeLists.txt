project(hermes_shm)



#------------------------------------------------------------------------------
# Test Cases
#------------------------------------------------------------------------------
set (LIBS
        hermes_shm_host
        Catch2::Catch2
        ${MPI_LIBS}
        ${OpenMP_LIBS})
add_executable(test_cereal_exec
        ${TEST_MAIN}/main.cc
        test_init.cc
        test_cereal.cc)
add_dependencies(test_cereal_exec
        hermes_shm_host)
target_link_libraries(test_cereal_exec ${LIBS})

add_test(NAME test_cereal COMMAND ${CMAKE_BINARY_DIR}/bin/test_cereal_exec)

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_cereal_exec
        LIBRARY DESTINATION ${HSHM_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${HSHM_INSTALL_LIB_DIR}
        RUNTIME DESTINATION ${HSHM_INSTALL_BIN_DIR})

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HSHM_ENABLE_COVERAGE)
    set_coverage_flags(test_cereal_exec)
endif()