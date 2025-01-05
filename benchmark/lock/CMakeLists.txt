cmake_minimum_required(VERSION 3.10)
project(hermes_shm)

set(CMAKE_CXX_STANDARD 17)

if (HERMES_ENABLE_OPENMP)
    include_directories( ${Boost_INCLUDE_DIRS} )
    include_directories( ${TEST_MAIN} )
    add_executable(benchmark_lock
        ${TEST_MAIN}/main.cc
        test_init.cc
        benchmark_mutex.cc
    )
    add_dependencies(benchmark_lock hermes_shm_data_structures)
    target_link_libraries(benchmark_lock
            hermes_shm_data_structures
            Catch2::Catch2
            ${MPI_LIBS}
            ${OpenMP_LIBS})
endif()