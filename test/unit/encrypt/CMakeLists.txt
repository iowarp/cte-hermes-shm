project(hermes_shm)



#------------------------------------------------------------------------------
# Build Tests
#------------------------------------------------------------------------------
add_executable(test_encrypt_exec
        ${TEST_MAIN}/main.cc
        test_init.cc
        test_encrypt.cc)
add_dependencies(test_encrypt_exec hermes_shm_host)
target_link_libraries(test_encrypt_exec
        hermes_shm_host Catch2::Catch2)

add_test(NAME test_encrypt COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_encrypt_exec "~[error=FatalError]")

#-----------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_encrypt_exec
        LIBRARY DESTINATION ${HERMES_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${HERMES_INSTALL_LIB_DIR}
        RUNTIME DESTINATION ${HERMES_INSTALL_BIN_DIR})

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HERMES_ENABLE_COVERAGE)
    set_coverage_flags(test_encrypt_exec)
endif()