project(hermes_shm)

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH
    "/mnt/common/rpawar4/spack/opt/spack/linux-ubuntu22.04-skylake_avx512/gcc-11.4.0/mochi-thallium-0.11.3-jjyvysrt6svnlg4kgpwmwqrfei5rd4wl/lib"
    "/mnt/common/rpawar4/spack/opt/spack/linux-ubuntu22.04-skylake_avx512/gcc-11.4.0/mochi-margo-0.15.0-vavjqayvw6tdon75kwr35igrerz3a6c7/lib"
    "/mnt/common/rpawar4/spack/opt/spack/linux-ubuntu22.04-skylake_avx512/gcc-11.4.0/mercury-2.3.1-5q7rgtc7nph66b544cgmeeayybijjdh4/lib"
    "/mnt/common/rpawar4/spack/opt/spack/linux-ubuntu22.04-skylake_avx512/gcc-11.4.0/argobots-1.1-h4lbxil3vaiikyak3twhwysimyd7izr2/lib"
    "/mnt/common/rpawar4/spack/opt/spack/linux-ubuntu22.04-skylake_avx512/gcc-11.4.0/json-c-0.16-5nwshmp3jvwca2nwur3bvf7yn4qcnlno/lib"
)

# Find ZeroMQ for the test target
find_library(ZMQ_LIBRARY NAMES zmq
  HINTS /mnt/common/rpawar4/spack/opt/spack/linux-ubuntu22.04-skylake_avx512/gcc-11.4.0/libzmq-4.3.5-pok35prm6qglcrwqiiijyp6gkqndoqsi/lib
)
if(NOT ZMQ_LIBRARY)
  message(FATAL_ERROR "ZeroMQ not found for test! Please set the correct path.")
endif()
get_filename_component(ZMQ_LIB_DIR ${ZMQ_LIBRARY} DIRECTORY)
link_directories(${ZMQ_LIB_DIR})

# Find Thallium (try pkg-config first, then fallback to Spack location)
find_package(PkgConfig)
pkg_check_modules(THALLIUM thallium)

if(THALLIUM_FOUND)
  set(THALLIUM_INCLUDE_DIRS ${THALLIUM_INCLUDE_DIRS})
  set(THALLIUM_LIBRARIES ${THALLIUM_LIBRARIES})
  set(THALLIUM_LDFLAGS ${THALLIUM_LDFLAGS})
else()
  # Fallback: hardcode the Spack install path if pkg-config fails
  set(THALLIUM_ROOT "/mnt/common/rpawar4/spack/opt/spack/linux-ubuntu22.04-skylake_avx512/gcc-11.4.0/mochi-thallium-0.11.3-jjyvysrt6svnlg4kgpwmwqrfei5rd4wl")
  set(THALLIUM_INCLUDE_DIRS "${THALLIUM_ROOT}/include")
  find_library(THALLIUM_LIBRARIES NAMES thallium HINTS "${THALLIUM_ROOT}/lib")
  if(NOT THALLIUM_LIBRARIES)
    message(FATAL_ERROR "Thallium not found! Please check your installation.")
  endif()
  set(THALLIUM_LDFLAGS "")
endif()

add_subdirectory(unit)

# Simple unit tests (easier to understand)
# Commented out legacy tests that use the old API and do not build with the new header-only interface
# add_executable(simple_client_test simple_client_test.cc)
# target_include_directories(simple_client_test PRIVATE ${THALLIUM_INCLUDE_DIRS})
# target_link_libraries(simple_client_test PRIVATE lightbeam ${ZMQ_LIBRARY} pthread ${THALLIUM_LIBRARIES})
# if(THALLIUM_LDFLAGS)
#   separate_arguments(THALLIUM_LDFLAGS)
#   target_link_options(simple_client_test PRIVATE ${THALLIUM_LDFLAGS})
# endif()

# add_executable(simple_server_test simple_server_test.cc)
# target_include_directories(simple_server_test PRIVATE ${THALLIUM_INCLUDE_DIRS})
# target_link_libraries(simple_server_test PRIVATE lightbeam ${ZMQ_LIBRARY} pthread ${THALLIUM_LIBRARIES})
# if(THALLIUM_LDFLAGS)
#   target_link_options(simple_server_test PRIVATE ${THALLIUM_LDFLAGS})
# endif()

