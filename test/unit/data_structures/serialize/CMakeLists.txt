project(hermes_shm)



add_subdirectory(shm)

if (HSHM_RPC_THALLIUM)
    add_subdirectory(thallium)
endif()

if (HSHM_ENABLE_CEREAL)
    add_subdirectory(cereal)
endif()