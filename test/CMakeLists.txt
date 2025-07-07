project(hermes_shm)

add_subdirectory(unit)

add_executable(lightbeam_test lightbeam_test.cc)
target_link_libraries(lightbeam_test PRIVATE lightbeam zmq pthread)