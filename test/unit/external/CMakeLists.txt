project(hermes_shm)

# ------------------------------------------------------------------------------
# External libraries
# ------------------------------------------------------------------------------
find_package(HermesShm CONFIG REQUIRED)
message(STATUS "found hermes_shm at ${HermesShm_DIR}")

# ------------------------------------------------------------------------------
# Build
# ------------------------------------------------------------------------------
add_executable(test test.cc)
target_link_libraries(test hshm::cxx)

# hshm_enable_cuda(17)
# add_cuda_executable(test_cuda FALSE test_cuda.cc)
# target_link_libraries(test_cuda hshm::cudacxx)