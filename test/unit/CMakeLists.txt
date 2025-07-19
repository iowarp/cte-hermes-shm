project(hermes_shm)

set(TEST_MAIN ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(data_structures)
add_subdirectory(allocators)
add_subdirectory(allocators_mpi)
add_subdirectory(backend)
add_subdirectory(types)
add_subdirectory(thread)
add_subdirectory(singleton)

if(NOT HSHM_COMPILER_MSVC)
    add_subdirectory(interceptor)
endif()

if(HSHM_ENABLE_CUDA)
    add_subdirectory(cuda)
endif()

if(HSHM_ENABLE_ROCM)
    add_subdirectory(rocm)
endif()

if(HSHM_ENABLE_COMPRESS)
    add_subdirectory(compress)
endif()

if(HSHM_ENABLE_ENCRYPT)
    add_subdirectory(encrypt)
endif()

if(HSHM_ENABLE_ZMQ)
    add_subdirectory(lightbeam)
endif()