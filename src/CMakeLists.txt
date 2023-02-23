cmake_minimum_required(VERSION 3.10)
project(hermes_shm)

set(CMAKE_CXX_STANDARD 17)

##################Build hermes main packages

add_library(hermes_shm_data_structures
        thread/mutex.cc
        thread/rwlock.cc
        memory/malloc_allocator.cc
        memory/stack_allocator.cc
        memory/fixed_page_allocator.cc
        memory/memory_registry.cc
        memory/memory_manager.cc
        data_structure_singleton.cc)
target_link_libraries(hermes_shm_data_structures
        yaml-cpp pthread -lrt -ldl)

##################Install hermes
install(
        TARGETS hermes_shm_data_structures
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

##################COVERAGE
if(HERMES_ENABLE_COVERAGE)
    set_coverage_flags(hermes_shm_data_structures)
endif()