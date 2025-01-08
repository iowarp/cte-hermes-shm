cmake_minimum_required(VERSION 3.10)
project(hermes_shm)

add_library(my_lib1 STATIC my_lib1.cc)
add_library(my_lib2 STATIC my_lib2.cc)

add_executable(test_singleton ${TEST_MAIN}/main.cc test_init.cc test_singleton.cc)
add_dependencies(test_singleton my_lib1 my_lib2)
target_link_libraries(test_singleton PRIVATE my_lib1 my_lib2 Catch2::Catch2)