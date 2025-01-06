cmake_minimum_required(VERSION 3.10)
project(hermes_shm)



if (HERMES_ENABLE_OPENMP)
    include_directories( ${Boost_INCLUDE_DIRS} )
    include_directories( ${TEST_MAIN} )
    add_executable(benchmark_allocators_exec
            ${TEST_MAIN}/main.cc
            test_init.cc
            allocator.cc
    )
    add_dependencies(benchmark_allocators_exec hermes_shm_data_structures)
    target_link_libraries(benchmark_allocators_exec
            hermes_shm_data_structures
            Catch2::Catch2
            ${MPI_LIBS}
            ${OpenMP_LIBS}
            ${Boost_LIBRARIES})
endif()