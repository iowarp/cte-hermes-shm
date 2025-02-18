project(hermes_shm)

include_directories(${HSHM_ROOT}/test/unit)
add_subdirectory(containers)
add_subdirectory(containers_mpi)
add_subdirectory(serialize)
