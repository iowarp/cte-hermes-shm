//
// Created by llogan on 10/9/24.
//

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <stdio.h>

#include <cassert>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/data_structures/all.h"
#include "hermes_shm/types/argpack.h"
#include "hermes_shm/types/atomic.h"
#include "hermes_shm/util/singleton.h"

#define HSHM_DEFAULT_GPU_ALLOC_T hipc::ThreadLocalAllocator
HSHM_DATA_STRUCTURES_TEMPLATE_BASE(gpu::ipc, hshm::ipc,
                                   HSHM_DEFAULT_GPU_ALLOC_T)

HSHM_GPU_KERNEL void mpsc_kernel(gpu::ipc::mpsc_queue<int> *queue) {
  hipc::ScopedTlsAllocator<HSHM_DEFAULT_GPU_ALLOC_T> ctx_alloc(
      queue->GetCtxAllocator());
  queue->GetThreadLocal(ctx_alloc);
  queue->emplace(10);
}

hipc::AllocatorId alloc_id(1, 0);
hshm::chararr shm_url = "test_serializers";

template <typename BackendT>
HSHM_DEFAULT_GPU_ALLOC_T *CreateShmem() {
  auto mem_mngr = HSHM_MEMORY_MANAGER;
  mem_mngr->UnregisterAllocator(alloc_id);
  mem_mngr->DestroyBackend(hipc::MemoryBackendId::Get(0));
  mem_mngr->CreateBackend<BackendT>(hipc::MemoryBackendId::Get(0),
                                    MEGABYTES(100), shm_url, 0);
  auto *alloc = mem_mngr->CreateAllocator<HSHM_DEFAULT_GPU_ALLOC_T>(
      hipc::MemoryBackendId::Get(0), alloc_id,
      sizeof(gpu::ipc::mpsc_queue<int>));
  return alloc;
}

template <typename BackendT>
HSHM_DEFAULT_GPU_ALLOC_T *LoadShmem() {
  auto mem_mngr = HSHM_MEMORY_MANAGER;
  mem_mngr->AttachBackend(BackendT::EnumType, shm_url);
  auto *alloc = mem_mngr->GetAllocator<HSHM_DEFAULT_GPU_ALLOC_T>(alloc_id);
  return alloc;
}

template <typename BackendT>
void mpsc_test() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  HSHM_DEFAULT_GPU_ALLOC_T *alloc;
  if (rank == 0) {
    alloc = CreateShmem<BackendT>();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank != 0) {
    alloc = LoadShmem<BackendT>();
  }
  hipc::delay_ar<gpu::ipc::mpsc_queue<int>> &queue =
      *alloc->GetCustomHeader<hipc::delay_ar<gpu::ipc::mpsc_queue<int>>>();
  hipc::CtxAllocator<HSHM_DEFAULT_GPU_ALLOC_T> ctx_alloc(alloc);
  if (rank == 0) {
    HSHM_MAKE_AR(queue, alloc, 256 * 256);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    mpsc_kernel<<<16, 16>>>(queue.get());
    HIP_ERROR_CHECK(hipDeviceSynchronize());
  } else {
    while (queue->size() < 16 * 16) {
      continue;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  printf("SHARED MEMORY QUEUE WORKS: %d!\n", (int)queue->size());
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  mpsc_test<hipc::RocmMalloc>();
  MPI_Finalize();
}
