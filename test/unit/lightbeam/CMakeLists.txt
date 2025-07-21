find_package(PkgConfig REQUIRED)
pkg_check_modules(ZeroMQ REQUIRED libzmq)
pkg_check_modules(Thallium REQUIRED thallium)

if(HSHM_ENABLE_ZMQ)
    add_executable(lightbeam_transport_test lightbeam_transport_test.cc)
    target_include_directories(lightbeam_transport_test PRIVATE ${ZeroMQ_INCLUDE_DIRS} ${Thallium_INCLUDE_DIRS})
    target_link_directories(lightbeam_transport_test PRIVATE ${ZeroMQ_LIBRARY_DIRS} ${Thallium_LIBRARY_DIRS})
    target_link_libraries(lightbeam_transport_test ${ZeroMQ_LIBRARIES} ${Thallium_LIBRARIES})
    target_compile_definitions(lightbeam_transport_test PRIVATE HSHM_ENABLE_ZMQ)
    add_test(NAME lightbeam_transport_test COMMAND lightbeam_transport_test)
endif()

add_executable(distributed_lightbeam_test distributed_lightbeam_test.cc)
target_include_directories(distributed_lightbeam_test PRIVATE ${ZeroMQ_INCLUDE_DIRS} ${Thallium_INCLUDE_DIRS})
target_link_directories(distributed_lightbeam_test PRIVATE ${ZeroMQ_LIBRARY_DIRS} ${Thallium_LIBRARY_DIRS})
target_link_libraries(distributed_lightbeam_test ${ZeroMQ_LIBRARIES} ${Thallium_LIBRARIES})

