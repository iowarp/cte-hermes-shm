//
// Created by llogan on 10/9/24.
//

#include <hip/hip_runtime.h>
#include <stdio.h>

#include <cassert>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/data_structures/all.h"
#include "hermes_shm/data_structures/ipc/ring_queue.h"
#include "hermes_shm/data_structures/ipc/string.h"
#include "hermes_shm/data_structures/ipc/unordered_map.h"
#include "hermes_shm/memory/memory_manager.h"
#include "hermes_shm/thread/lock/mutex.h"
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

__global__ void backend_kernel(MyStruct *ptr) {
  // int idx = blockIdx.x * blockDim.x + threadIdx.x;
  MyStruct quest;
  ptr->x = quest.DoSomething();
  ptr->y =
      hshm::PassArgPack::Call(hshm::make_argpack(0, 1, 2),
                              [](int x, int y, int z) { return x + y + z; });
  ptr->x = 25;
}

void backend_test() {
  auto mem_mngr = HERMES_MEMORY_MANAGER;
  // Allocate memory on the host and device using UM
  size_t size = sizeof(MyStruct);

  // Create a MyStruct instance and copy it to both host and device memory
  hshm::ipc::RocmShmMmap shm;
  shm.shm_init(hipc::MemoryBackendId::Get(0), size, "shmem_test", 0);
  MyStruct *shm_struct = (MyStruct *)shm.data_;
  shm_struct->x = 10;
  shm_struct->y = 3.14f;

  // Launch a CUDA kernel that accesses the shared memory
  int blockSize = 256;
  int numBlocks = 1;
  dim3 block(blockSize);
  dim3 grid(numBlocks);
  backend_kernel<<<grid, block>>>(shm_struct);
  HIP_ERROR_CHECK(hipDeviceSynchronize());

  // Verify correctness
  MyStruct new_struct = *shm_struct;
  printf("Result: x=%d, y=%f\n", new_struct.x, new_struct.y);
  assert(new_struct.x == 25);
  assert(new_struct.y == 3);

  // Free memory
  shm.shm_destroy();
}

__global__ void singleton_kernel_p1() {
  *hshm::EasyLockfreeSingleton<int>::GetInstance() = 25;
}

__global__ void singleton_kernel(MyStruct *ptr) {
  ptr->x = *hshm::EasyLockfreeSingleton<int>::GetInstance();
  ptr->y = 3;
}

void singleton_test() {
  auto mem_mngr = HERMES_MEMORY_MANAGER;
  // Allocate memory on the host and device using UM
  size_t size = sizeof(MyStruct);

  // Create a MyStruct instance and copy it to both host and device memory
  hshm::ipc::RocmShmMmap shm;
  shm.shm_init(hipc::MemoryBackendId::Get(0), size, "shmem_test", 0);
  MyStruct *shm_struct = (MyStruct *)shm.data_;
  shm_struct->x = 10;
  shm_struct->y = 3.14f;

  // Launch a CUDA kernel that accesses the shared memory
  int blockSize = 256;
  int numBlocks = 1;
  dim3 block(blockSize);
  dim3 grid(numBlocks);
  singleton_kernel_p1<<<1, 1>>>();
  singleton_kernel<<<grid, block>>>(shm_struct);
  HIP_ERROR_CHECK(hipDeviceSynchronize());

  // Verify correctness
  MyStruct new_struct = *shm_struct;
  printf("Result: x=%d, y=%f\n", new_struct.x, new_struct.y);
  assert(new_struct.x == 25);
  assert(new_struct.y == 3);

  // Free memory
  shm.shm_destroy();
}

__global__ void mpsc_kernel(gpu::ipc::mpsc_queue<int> *queue) {
  hipc::ScopedTlsAllocator<HSHM_DEFAULT_GPU_ALLOC_T> ctx_alloc(
      queue->GetCtxAllocator());
  queue->GetThreadLocal(ctx_alloc);
  queue->emplace(10);
}

void mpsc_test() {
  hshm::chararr shm_url = "test_serializers";
  hipc::AllocatorId alloc_id(0, 1);
  auto mem_mngr = HERMES_MEMORY_MANAGER;
  mem_mngr->UnregisterAllocator(alloc_id);
  mem_mngr->DestroyBackend(hipc::MemoryBackendId::Get(0));
  mem_mngr->CreateBackend<hipc::RocmShmMmap>(hipc::MemoryBackendId::Get(0),
                                             MEGABYTES(100), shm_url, 0);
  auto *alloc = mem_mngr->CreateAllocator<HSHM_DEFAULT_GPU_ALLOC_T>(
      hipc::MemoryBackendId::Get(0), alloc_id, 0);
  hipc::CtxAllocator<HSHM_DEFAULT_GPU_ALLOC_T> ctx_alloc(alloc);
  auto *queue =
      ctx_alloc->NewObj<gpu::ipc::mpsc_queue<int>>(ctx_alloc.ctx_, 256 * 256);
  printf("GetSize: %lu\n", queue->GetSize());
  mpsc_kernel<<<16, 16>>>(queue);
  HIP_ERROR_CHECK(hipDeviceSynchronize());
  printf("GetSize: %lu\n", queue->GetSize());
  int val, sum = 0;
  while (!queue->pop(val).IsNull()) {
    sum += val;
  }
  printf("SUM: %d\n", sum);
}

__global__ void atomic_kernel(hipc::rocm_atomic<hshm::min_u64> *x) {
  x->fetch_add(1);
}

void atomic_test() {
  hipc::rocm_atomic<hshm::min_u64> *x;
  HIP_ERROR_CHECK(hipDeviceSynchronize());
  HIP_ERROR_CHECK(hipSetDevice(0));
  size_t size = sizeof(hipc::rocm_atomic<hshm::min_u64>);
  HIP_ERROR_CHECK(hipHostAlloc(&x, size, hipHostAllocMapped));
  atomic_kernel<<<64, 64>>>(x);
  HIP_ERROR_CHECK(hipDeviceSynchronize());
  HILOG(kInfo, "ATOMIC: {}", x->load());
  assert(x->load() == 64 * 64);
}

int main() {
  // atomic_test();
  // singleton_test();
  // backend_test();
  mpsc_test();
}