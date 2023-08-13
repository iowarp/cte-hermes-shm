cmake_minimum_required(VERSION 3.10)
project(hermes_shm)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(shm)

if (HERMES_RPC_THALLIUM)
    add_subdirectory(thallium)
endif()

if (HERMES_ENABLE_CEREAL)
    add_subdirectory(cereal)
endif()