project(hermes_shm)



if (HSHM_ENABLE_OPENMP)
    include_directories( ${Boost_INCLUDE_DIRS} )
    include_directories( ${TEST_MAIN} )
    add_executable(benchmark_data_structures_exec
            ${TEST_MAIN}/main.cc
            test_init.cc
            atomic.cc
            ref.cc
            string.cc
            list.cc
            vector.cc
            unordered_map.cc
            queue.cc
            lock.cc
    )
    add_dependencies(benchmark_data_structures_exec hermes_shm_host)
    target_link_libraries(benchmark_data_structures_exec
            hermes_shm_host
            Catch2::Catch2
            ${OpenMP_LIBS}
            ${Boost_LIBRARIES})
endif()
