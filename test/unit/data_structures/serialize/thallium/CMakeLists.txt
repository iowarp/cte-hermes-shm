cmake_minimum_required(VERSION 3.10)
project(hermes_shm)

set(CMAKE_CXX_STANDARD 17)

add_executable(test_thallium
        ${TEST_MAIN}/main.cc
        test_init.cc
        thallium.cc)
add_dependencies(test_thallium
        hermes_shm_data_structures)
target_link_libraries(test_thallium
        hermes_shm_data_structures
        Catch2::Catch2
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        thallium)

add_test(NAME test_thallium COMMAND
        ${CMAKE_CURRENT_BINARY_DIR}/test_thallium)

