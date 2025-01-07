cmake_minimum_required(VERSION 3.10)
project(hermes_shm)



add_subdirectory(shm)

if (HERMES_RPC_THALLIUM)
    add_subdirectory(thallium)
endif()

if (HERMES_ENABLE_CEREAL)
    add_subdirectory(cereal)
endif()