project(hermes_shm)



if (HSHM_ENABLE_OPENMP)
    include_directories( ${Boost_INCLUDE_DIRS} )
    include_directories( ${TEST_MAIN} )
    add_executable(benchmark_lock
        ${TEST_MAIN}/main.cc
        test_init.cc
        benchmark_mutex.cc
    )
    add_dependencies(benchmark_lock hermes_shm_host)
    target_link_libraries(benchmark_lock
            hermes_shm_host
            Catch2::Catch2
            ${MPI_LIBS}
            ${OpenMP_LIBS})
endif()