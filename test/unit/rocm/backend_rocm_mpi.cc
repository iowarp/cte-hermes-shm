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
#include "hermes_shm/util/logging.h"
#include "hermes_shm/util/singleton.h"

typedef hipc::GpuStackAllocator AllocT;
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

HSHM_GPU_KERNEL void vector_add_kernel(float *a, float *b, float *c, int n) {
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
  if (rank == 0) {
    alloc = CreateShmem<BackendT>();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank != 0) {
    alloc = LoadShmem<BackendT>();
  }
  Header *header = alloc->GetCustomHeader<Header>();
  hipc::delay_ar<gpu::ipc::mpsc_queue<int>> &queue = header->queue_;
  hipc::CtxAllocator<AllocT> ctx_alloc(alloc);
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

template <typename BackendT>
void alloc_test() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  AllocT *alloc;
  HILOG(kInfo, "Creating shared memory allocator: {}", alloc_id);
  if (rank == 0) {
    alloc = CreateShmem<BackendT>();
  }
  HILOG(kInfo, "Created shared memory allocator: {}", alloc_id);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank != 0) {
    alloc = LoadShmem<BackendT>();
  }
  Header *header = alloc->GetCustomHeader<Header>();
  size_t block = 256;
  size_t size = 256 * 256;
  HILOG(kInfo, "Beginning to allocate: {}", alloc_id);
  if (rank == 0) {
    header->a = alloc->Allocate<hipc::Pointer>(HSHM_MCTX, size);
    header->b = alloc->Allocate<hipc::Pointer>(HSHM_MCTX, size);
    header->c = alloc->Allocate<hipc::Pointer>(HSHM_MCTX, size);
  }
  hipc::FullPtr<float> a(header->a);
  hipc::FullPtr<float> b(header->b);
  hipc::FullPtr<float> c(header->c);
  MPI_Barrier(MPI_COMM_WORLD);
  vector_add_kernel<<<block, block>>>(a.ptr_, b.ptr_, c.ptr_, size);
  HIP_ERROR_CHECK(hipDeviceSynchronize());
  // float sum = 0;
  // for (size_t i = 0; i < size; ++i) {
  //   sum += header->c.ptr_[i];
  // }
  // HILOG(kInfo, "SUM: {}", sum);
  MPI_Barrier(MPI_COMM_WORLD);
  HILOG(kInfo, "Finished", alloc_id);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  alloc_test<hipc::RocmMalloc>();
  // mpsc_test<hipc::RocmShmMmap>();
  MPI_Finalize();
}
