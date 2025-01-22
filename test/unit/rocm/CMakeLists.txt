project(hermes_shm_cuda LANGUAGES C CXX)

# ------------------------------------------------------------------------------
# Build Tests
# ------------------------------------------------------------------------------
set(TEST_ROCM_TARGETS)
add_rocm_gpu_executable(test_memory_rocm_exec FALSE "backend_rocm.cc")
add_dependencies(test_memory_rocm_exec
        hermes_shm_rocm_gpu)
target_link_libraries(test_memory_rocm_exec PUBLIC hermes_shm_rocm_gpu)
list(APPEND TEST_ROCM_TARGETS test_memory_rocm_exec)

if(HSHM_ENABLE_MPI)
        add_rocm_gpu_executable(test_memory_rocm_exec_mpi FALSE "backend_rocm_mpi.cc")
        add_dependencies(test_memory_rocm_exec_mpi
                hermes_shm_rocm_gpu)
        target_link_libraries(test_memory_rocm_exec_mpi PUBLIC hermes_shm_rocm_gpu ${MPI_LIBS})
        list(APPEND TEST_ROCM_TARGETS test_memory_rocm_exec_mpi)
endif()

# ------------------------------------------------------------------------------
# Install Targets
# ------------------------------------------------------------------------------
install(TARGETS
        ${TEST_ROCM_TARGETS}
        LIBRARY DESTINATION ${HSHM_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${HSHM_INSTALL_LIB_DIR}
        RUNTIME DESTINATION ${HSHM_INSTALL_BIN_DIR})

# -----------------------------------------------------------------------------
# Coverage
# -----------------------------------------------------------------------------
if(HSHM_ENABLE_COVERAGE)
        set_coverage_flags(test_memory_rocm_exec)
endif()