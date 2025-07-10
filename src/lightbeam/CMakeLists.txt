# Find ZeroMQ
find_path(ZMQ_INCLUDE_DIR zmq.h
  HINTS /mnt/common/rpawar4/spack/opt/spack/linux-ubuntu22.04-skylake_avx512/gcc-11.4.0/libzmq-4.3.5-pok35prm6qglcrwqiiijyp6gkqndoqsi/include
)
find_library(ZMQ_LIBRARY NAMES zmq
  HINTS /mnt/common/rpawar4/spack/opt/spack/linux-ubuntu22.04-skylake_avx512/gcc-11.4.0/libzmq-4.3.5-pok35prm6qglcrwqiiijyp6gkqndoqsi/lib
)

if(NOT ZMQ_INCLUDE_DIR OR NOT ZMQ_LIBRARY)
  message(FATAL_ERROR "ZeroMQ not found! Please set the correct path.")
endif()

# Find Libfabric
find_path(LIBFABRIC_INCLUDE_DIR rdma/fabric.h
  HINTS /mnt/common/rpawar4/spack/opt/spack/linux-ubuntu22.04-skylake_avx512/gcc-11.4.0/libfabric-1.21.0-oisnk33gk3t6zirzelipj4ku2ieegzv4/include
)
find_library(LIBFABRIC_LIBRARY NAMES fabric
  HINTS /mnt/common/rpawar4/spack/opt/spack/linux-ubuntu22.04-skylake_avx512/gcc-11.4.0/libfabric-1.21.0-oisnk33gk3t6zirzelipj4ku2ieegzv4/lib
)

if(NOT LIBFABRIC_INCLUDE_DIR OR NOT LIBFABRIC_LIBRARY)
  message(FATAL_ERROR "Libfabric not found! Please set the correct path.")
endif()

# Add the library directory for the linker
get_filename_component(ZMQ_LIB_DIR ${ZMQ_LIBRARY} DIRECTORY)
get_filename_component(LIBFABRIC_LIB_DIR ${LIBFABRIC_LIBRARY} DIRECTORY)
link_directories(${ZMQ_LIB_DIR} ${LIBFABRIC_LIB_DIR})

add_library(lightbeam STATIC
    lightbeam.cc
    server.cc
    client.cc
    zmq_client.cc
    zmq_server.cc
    libfabric_client.cc
    libfabric_server.cc
    libfabric_common.cc
    utils.cc
    transport_factory.cc
)

target_include_directories(lightbeam PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/hermes_shm/lightbeam
    ${ZMQ_INCLUDE_DIR}
    ${LIBFABRIC_INCLUDE_DIR}
)

target_link_libraries(lightbeam PUBLIC ${ZMQ_LIBRARY} ${LIBFABRIC_LIBRARY}) 