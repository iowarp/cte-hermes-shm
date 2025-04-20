//
// Created by llogan on 10/9/24.
//

#include <cuda_runtime.h>
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

struct MyStruct {
  int x;
  float y;

  __host__ __device__ int DoSomething() {
#ifdef HSHM_IS_GPU
    return 25;
#else
    return 10;
#endif
  }
};

HSHM_GPU_KERNEL void backend_kernel(MyStruct *ptr) {
  // int idx = blockIdx.x * blockDim.x + threadIdx.x;
  MyStruct quest;
  ptr->x = quest.DoSomething();
  ptr->y =
      hshm::PassArgPack::Call(hshm::make_argpack(0, 1, 2),
                              [](int x, int y, int z) { return x + y + z; });
  *hshm::LockfreeCrossSingleton<int>::GetInstance() = 25;
  ptr->x = *hshm::LockfreeCrossSingleton<int>::GetInstance();
}

void backend_test() {
  auto mem_mngr = HSHM_MEMORY_MANAGER;
  // Allocate memory on the host and device using UM
  size_t size = sizeof(MyStruct);

  // Create a MyStruct instance and copy it to both host and device memory
  hshm::ipc::GpuShmMmap shm;
  shm.shm_init(hipc::MemoryBackendId::Get(0), size, "shmem_test");
  MyStruct *shm_struct = (MyStruct *)shm.data_;
  shm_struct->x = 10;
  shm_struct->y = 3.14f;

  // Launch a CUDA kernel that accesses the shared memory
  int blockSize = 256;
  int numBlocks = 1;
  dim3 block(blockSize);
  dim3 grid(numBlocks);
  backend_kernel<<<grid, block>>>(shm_struct);
  cudaDeviceSynchronize();

  // Verify correctness
  MyStruct new_struct = *shm_struct;
  printf("Result: x=%d, y=%f\n", new_struct.x, new_struct.y);
  assert(new_struct.x == 25);
  assert(new_struct.y == 3);

  // Free memory
  shm.shm_destroy();
}

HSHM_GPU_KERNEL void mpsc_kernel(gpu::ipc::mpsc_queue<int> *queue) {
  hipc::ScopedTlsAllocator<HSHM_DEFAULT_GPU_ALLOC_T> ctx_alloc(
      queue->GetCtxAllocator());
  queue->GetThreadLocal(ctx_alloc);
  queue->emplace(10);
}

void mpsc_test() {
  hshm::chararr shm_url = "test_serializers";
  hipc::AllocatorId alloc_id(1, 0);
  auto mem_mngr = HSHM_MEMORY_MANAGER;
  mem_mngr->UnregisterAllocator(alloc_id);
  mem_mngr->DestroyBackend(hipc::MemoryBackendId::Get(0));
  mem_mngr->CreateBackend<hipc::GpuShmMmap>(hipc::MemoryBackendId::Get(0),
                                            hshm::Unit<size_t>::Megabytes(100),
                                            shm_url);
  auto *alloc = mem_mngr->CreateAllocator<HSHM_DEFAULT_GPU_ALLOC_T>(
      hipc::MemoryBackendId::Get(0), alloc_id, 0);
  // mem_mngr->ScanBackends();
  hipc::CtxAllocator<HSHM_DEFAULT_GPU_ALLOC_T> ctx_alloc(alloc);
  auto *queue =
      ctx_alloc->NewObj<gpu::ipc::mpsc_queue<int>>(ctx_alloc.ctx_, 256 * 256);
  printf("GetSize: %lu\n", (long unsigned)queue->GetSize());
  mpsc_kernel<<<1, 1>>>(queue);
  cudaDeviceSynchronize();
  printf("GetSize: %lu\n", (long unsigned)queue->GetSize());
  int val, sum = 0;
  while (!queue->pop(val).IsNull()) {
    sum += val;
  }
  printf("SUM: %d\n", sum);
}

HSHM_GPU_KERNEL void atomic_kernel(hipc::atomic<hshm::size_t> *x) {
  x->fetch_add(1);
}

void atomic_test() {
  hipc::atomic<hshm::size_t> *x;
  cudaDeviceSynchronize();
  cudaSetDevice(0);
  size_t size = sizeof(hipc::atomic<hshm::size_t>);
  cudaHostAlloc(&x, size, cudaHostAllocMapped);
  atomic_kernel<<<64, 64>>>(x);
  cudaDeviceSynchronize();
  printf("ATOMIC: %llu\n", x->load());
}

int main() {
  mpsc_test();
  // atomic_test();
}
