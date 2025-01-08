cmake_minimum_required(VERSION 3.10)
project(hermes_shm)

add_library(singleton_lib STATIC singleton_lib.cc)
add_library(my_lib1 STATIC my_lib1.cc)
add_library(my_lib2 STATIC my_lib2.cc)

add_dependencies(my_lib1 singleton_lib)
add_dependencies(my_lib2 singleton_lib)

target_link_libraries(my_lib1 PRIVATE singleton_lib)
target_link_libraries(my_lib2 PRIVATE singleton_lib)

add_executable(test_singleton ${TEST_MAIN}/main.cc test_init.cc test_singleton.cc)
add_dependencies(test_singleton my_lib1 my_lib2 singleton_lib)
target_link_libraries(test_singleton PRIVATE my_lib1 my_lib2 singleton_lib Catch2::Catch2)