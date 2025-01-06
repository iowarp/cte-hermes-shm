cmake_minimum_required(VERSION 3.10)
project(hermes_shm)



include_directories(${HERMES_SHM_ROOT}/test/unit)
add_subdirectory(containers)
add_subdirectory(containers_mpi)
add_subdirectory(serialize)
