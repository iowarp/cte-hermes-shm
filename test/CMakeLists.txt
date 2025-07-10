project(hermes_shm)

# Find ZeroMQ for the test target
find_library(ZMQ_LIBRARY NAMES zmq
  HINTS /mnt/common/rpawar4/spack/opt/spack/linux-ubuntu22.04-skylake_avx512/gcc-11.4.0/libzmq-4.3.5-pok35prm6qglcrwqiiijyp6gkqndoqsi/lib
)
if(NOT ZMQ_LIBRARY)
  message(FATAL_ERROR "ZeroMQ not found for test! Please set the correct path.")
endif()
get_filename_component(ZMQ_LIB_DIR ${ZMQ_LIBRARY} DIRECTORY)
link_directories(${ZMQ_LIB_DIR})

add_subdirectory(unit)

add_executable(lightbeam_test lightbeam_test.cc)
target_link_libraries(lightbeam_test PRIVATE lightbeam ${ZMQ_LIBRARY} pthread)

# Separate client and server tests for benchmark purposes
add_executable(lightbeam_client_test lightbeam_client_test.cc)
target_link_libraries(lightbeam_client_test PRIVATE lightbeam ${ZMQ_LIBRARY} pthread)

add_executable(lightbeam_server_test lightbeam_server_test.cc)
target_link_libraries(lightbeam_server_test PRIVATE lightbeam ${ZMQ_LIBRARY} pthread)