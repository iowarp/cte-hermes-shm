project(hermes_shm)

add_subdirectory(shm)

if(HSHM_ENABLE_CEREAL)
    add_subdirectory(cereal)
endif()