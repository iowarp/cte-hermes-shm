//
// Created by llogan on 10/9/24.
//

#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>

#include <cassert>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/data_structures/all.h"
#include "hermes_shm/types/argpack.h"
#include "hermes_shm/types/atomic.h"
#include "hermes_shm/util/gpu_api.h"
#include "hermes_shm/util/logging.h"
#include "hermes_shm/util/singleton.h"

#define ROOT_RANK 0

// USE WHEN GpuMalloc
typedef hipc::GpuStackAllocator AllocT;
// USE WHEN GpuShm
// typedef hipc::StackAllocator AllocT;
HSHM_DATA_STRUCTURES_TEMPLATE_BASE(gpu::ipc, hshm::ipc, AllocT)

struct Header {
  hipc::delay_ar<gpu::ipc::mpsc_queue<int>> queue_;
  hipc::Pointer a, b, c;
};

HSHM_GPU_KERNEL void mpsc_kernel(gpu::ipc::mpsc_queue<int> *queue) {
  hipc::ScopedTlsAllocator<AllocT> ctx_alloc(queue->GetCtxAllocator());
  queue->GetThreadLocal(ctx_alloc);
  queue->emplace(10);
}

HSHM_GPU_KERNEL void vector_add_kernel(char *a, char *b, char *c, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

hipc::AllocatorId alloc_id(1, 0);
hshm::chararr shm_url = "test_serializers";

template <typename BackendT>
AllocT *CreateShmem() {
  auto mem_mngr = HSHM_MEMORY_MANAGER;
  mem_mngr->UnregisterAllocator(alloc_id);
  mem_mngr->DestroyBackend(hipc::MemoryBackendId::Get(0));
  mem_mngr->CreateBackend<BackendT>(hipc::MemoryBackendId::Get(0),
                                    MEGABYTES(100), shm_url, 0);
  HILOG(kInfo, "Starting create: {}", alloc_id);
  auto *alloc = mem_mngr->CreateAllocator<AllocT>(hipc::MemoryBackendId::Get(0),
                                                  alloc_id, sizeof(Header));
  HILOG(kInfo, "Finished create: {}", alloc_id);
  return alloc;
}

template <typename BackendT>
AllocT *LoadShmem() {
  auto mem_mngr = HSHM_MEMORY_MANAGER;
  mem_mngr->AttachBackend(BackendT::EnumType, shm_url);
  auto *alloc = mem_mngr->GetAllocator<AllocT>(alloc_id);
  HILOG(kInfo, "Loading shared memory allocator: {}", alloc_id);
  return alloc;
}

template <typename BackendT>
void mpsc_test() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  AllocT *alloc;
  if (rank == ROOT_RANK) {
    alloc = CreateShmem<BackendT>();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank != ROOT_RANK) {
    alloc = LoadShmem<BackendT>();
  }
  Header *header = alloc->GetCustomHeader<Header>();
  hipc::delay_ar<gpu::ipc::mpsc_queue<int>> &queue = header->queue_;
  hipc::CtxAllocator<AllocT> ctx_alloc(alloc);
  if (rank == ROOT_RANK) {
    HSHM_MAKE_AR(queue, alloc, 256 * 256);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == ROOT_RANK) {
    mpsc_kernel<<<16, 16>>>(queue.get());
    hshm::GpuApi::Synchronize();
  } else {
    while (queue->size() < 16 * 16) {
      continue;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  printf("SHARED MEMORY QUEUE WORKS: %d!\n", (int)queue->size());
}

template <typename BackendT>
void alloc_test() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  AllocT *alloc;
  HILOG(kInfo, "Creating shared memory allocator: {}", alloc_id);
  if (rank == ROOT_RANK) {
    alloc = CreateShmem<BackendT>();
  }
  HILOG(kInfo, "Created shared memory allocator: {}", alloc_id);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank != ROOT_RANK) {
    alloc = LoadShmem<BackendT>();
  }
  Header *header = alloc->GetCustomHeader<Header>();
  size_t block = 256;
  size_t count = block * block;
  size_t size = count * sizeof(char);
  HILOG(kInfo, "Beginning to allocate: {}", alloc_id);
  if (rank == ROOT_RANK) {
    header->a = alloc->Allocate<hipc::Pointer>(HSHM_MCTX, size);
    header->b = alloc->Allocate<hipc::Pointer>(HSHM_MCTX, size);
    header->c = alloc->Allocate<hipc::Pointer>(HSHM_MCTX, size);
  }
  hipc::FullPtr<char> a(header->a);
  hipc::FullPtr<char> b(header->b);
  hipc::FullPtr<char> c(header->c);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank != ROOT_RANK) {
    hshm::GpuApi::Memset(a.ptr_, 1, size);
    hshm::GpuApi::Memset(b.ptr_, 2, size);
    hshm::GpuApi::Memset(c.ptr_, 0, size);
    vector_add_kernel<<<block, block>>>(a.ptr_, b.ptr_, c.ptr_, count);
    hshm::GpuApi::Synchronize();
    float sum = 0;
    std::vector<char> data(count);
    hshm::GpuApi::Memcpy(data.data(), c.ptr_, size);
    for (size_t i = 0; i < size; ++i) {
      sum += data[i];
    }
    HILOG(kInfo, "SUM: {}", sum);
    assert(sum == 3 * count);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  HILOG(kInfo, "Finished", alloc_id);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  alloc_test<hipc::GpuMalloc>();
  // alloc_test<hipc::GpuShmMmap>();
  // mpsc_test<hipc::GpuShmMmap>();
  MPI_Finalize();
}
