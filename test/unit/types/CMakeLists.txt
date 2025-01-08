cmake_minimum_required(VERSION 3.10)
project(hermes_shm)

#------------------------------------------------------------------------------
# Build Tests
#------------------------------------------------------------------------------
add_executable(test_types_exec
        ${TEST_MAIN}/main.cc
        test_init.cc
        test_argpack.cc
        test_util.cc
        test_atomics.cc)
add_dependencies(test_types_exec hermes_shm_host)
target_link_libraries(test_types_exec
        hermes_shm_host Catch2::Catch2 ${MPI_LIBS} ${OpenMP_LIBS})

add_test(NAME test_types COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_types_exec "~[error=FatalError]")
add_test(NAME test_fatal_logger
        COMMAND ${CMAKE_BINARY_DIR}/bin/test_types_exec "[error=FatalError]")
set_tests_properties(
        test_fatal_logger PROPERTIES
        WILL_FAIL TRUE)
set(INSTALLS ${INSTALLS} test_types_exec)

#------------------------------------------------------------------------------
# Build Timer Tests
#------------------------------------------------------------------------------
if (HERMES_ENABLE_MPI AND HERMES_ENABLE_OPENMP)
    add_executable(test_timers_exec
            ${TEST_MAIN}/main_mpi.cc
            test_init.cc
            test_timers.cc)
    add_dependencies(test_timers_exec hermes_shm_host)
    target_link_libraries(test_timers_exec
            hermes_shm_host Catch2::Catch2 ${MPI_LIBS} ${OpenMP_LIBS})

    add_test(NAME test_timers COMMAND
            ${CMAKE_BINARY_DIR}/bin/test_timers_exec)
    set(INSTALLS ${INSTALLS} test_timers_exec)
endif()

#-----------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        ${INSTALLS}
        LIBRARY DESTINATION ${HERMES_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${HERMES_INSTALL_LIB_DIR}
        RUNTIME DESTINATION ${HERMES_INSTALL_BIN_DIR})

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HERMES_ENABLE_COVERAGE)
    set_coverage_flags(test_types_exec)
    set_coverage_flags(test_timers_exec)
endif()