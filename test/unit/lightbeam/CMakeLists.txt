find_package(PkgConfig REQUIRED)
pkg_check_modules(ZeroMQ REQUIRED libzmq)
pkg_check_modules(Thallium REQUIRED thallium)

if(HSHM_ENABLE_ZMQ)
    add_executable(zmq_lightbeam_test zmq_lightbeam_test.cc)
    target_include_directories(zmq_lightbeam_test PRIVATE ${ZeroMQ_INCLUDE_DIRS} ${Thallium_INCLUDE_DIRS})
    target_link_directories(zmq_lightbeam_test PRIVATE ${ZeroMQ_LIBRARY_DIRS} ${Thallium_LIBRARY_DIRS})
    target_link_libraries(zmq_lightbeam_test ${ZeroMQ_LIBRARIES} ${Thallium_LIBRARIES})
endif()

add_executable(simple_thallium_test simple_thallium_test.cc)
target_include_directories(simple_thallium_test PRIVATE ${Thallium_INCLUDE_DIRS})
target_link_directories(simple_thallium_test PRIVATE ${Thallium_LIBRARY_DIRS})
target_link_libraries(simple_thallium_test ${Thallium_LIBRARIES})

add_executable(hostfile_test hostfile_test.cc)

add_executable(minimal_test minimal_test.cc) 