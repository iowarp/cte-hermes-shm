project(hermes_shm)



#------------------------------------------------------------------------------
# Build Tests
#------------------------------------------------------------------------------
set (LIBS
        hermes_shm_host
        Catch2::Catch2
        ${MPI_LIBS}
        ${OpenMP_LIBS}
        thallium)

add_executable(test_thallium_server server.cc)
add_dependencies(test_thallium_server
        hermes_shm_host)
target_link_libraries(test_thallium_server ${LIBS})
target_compile_definitions(test_thallium_server PRIVATE
        HSHM_DEFAULT_THREAD_MODEL=hshm::thread::Argobots)

#------------------------------------------------------------------------------
# Test Cases
#------------------------------------------------------------------------------
add_executable(test_thallium_exec
        ${TEST_MAIN}/main.cc
        test_init.cc
        test_thallium.cc)
add_dependencies(test_thallium_exec
        hermes_shm_host)
target_link_libraries(test_thallium_exec ${LIBS})
target_compile_definitions(test_thallium_server PRIVATE
        HSHM_DEFAULT_THREAD_MODEL=hshm::thread::Argobots)

add_test(NAME test_thallium COMMAND
        bash ${CMAKE_CURRENT_SOURCE_DIR}/test_thallium.sh
        "${CMAKE_BINARY_DIR}")

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_thallium_server
        test_thallium_exec
        LIBRARY DESTINATION ${HERMES_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${HERMES_INSTALL_LIB_DIR}
        RUNTIME DESTINATION ${HERMES_INSTALL_BIN_DIR})

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HERMES_ENABLE_COVERAGE)
    set_coverage_flags(test_thallium_server)
    set_coverage_flags(test_thallium_exec)
endif()