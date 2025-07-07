add_library(lightbeam STATIC lightbeam.cc)

target_include_directories(lightbeam PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/hermes_shm/lightbeam
)

target_link_libraries(lightbeam PUBLIC zmq) 