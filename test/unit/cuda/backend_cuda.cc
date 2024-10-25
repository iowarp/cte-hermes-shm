//
// Created by llogan on 10/9/24.
//

#include <cuda_runtime.h>
#include <stdio.h>
#include "hermes_shm/memory/backend/cuda_shm_mmap.h"
#include "hermes_shm/constants/macros.h"
#include "hermes_shm/types/argpack.h"
#include "hermes_shm/util/singleton/_easy_lockfree_singleton.h"
#include "hermes_shm/util/singleton/_global_singleton.h"
#include "hermes_shm/types/atomic.h"
#include "hermes_shm/thread/lock/mutex.h"
#include "hermes_shm/memory/memory_manager.h"
#include <cassert>
#include <hermes_shm/data_structures/ipc/mpsc_queue.h>

enum class TestMode {
  kWrite
};

struct MyStruct {
  int x;
  float y;

  __host__ __device__ int DoSomething() {
#ifdef __CUDA_ARCH__
    return 25;
#else
    return 10;
#endif
  }
};

__global__ void my_backend(MyStruct* ptr) {
  // int idx = blockIdx.x * blockDim.x + threadIdx.x;
  MyStruct quest;
  ptr->x = quest.DoSomething();
  ptr->y = hshm::PassArgPack::Call(
      hshm::make_argpack(0, 1, 2),
      [](int x, int y, int z) {
        return x + y + z;
      });
  *hshm::EasyLockfreeSingleton<int>::GetInstance() = 25;
  ptr->x = *hshm::EasyLockfreeSingleton<int>::GetInstance();
}

void backend_test() {
  auto mem_mngr = HERMES_MEMORY_MANAGER;
  // Allocate memory on the host and device using UM
  size_t size = sizeof(MyStruct);

  // Create a MyStruct instance and copy it to both host and device memory
  hshm::ipc::CudaShmMmap shm;
  shm.shm_init(size, "shmem_test", 0);
  MyStruct* shm_struct = (MyStruct*)shm.data_;
  shm_struct->x = 10;
  shm_struct->y = 3.14f;

  // Launch a CUDA kernel that accesses the shared memory
  int blockSize = 256;
  int numBlocks = 1;
  dim3 block(blockSize);
  dim3 grid(numBlocks);
  my_backend<<<grid, block>>>(shm_struct);
  cudaDeviceSynchronize();

  // Verify correctness
  MyStruct new_struct = *shm_struct;
  printf("Result: x=%d, y=%f\n", new_struct.x, new_struct.y);
  assert(new_struct.x == 25);
  assert(new_struct.y == 3);

  // Free memory
  shm.shm_destroy();
}

__global__ void my_allocator(hipc::MemoryBackend *backend,
                             hipc::Allocator *allocator) {
  auto mem_mngr = HERMES_MEMORY_MANAGER;
  mem_mngr->AttachBackend(backend);
  mem_mngr->AttachAllocator(allocator);
  hipc::uptr<hipc::vector<int>> vec = hipc::make_uptr<hipc::vector<int>>(10);
  for (int i = 0; i < 10; ++i) {
    (*vec)[i] = 10;
  }
}

void allocator_test() {
  auto mem_mngr = HERMES_MEMORY_MANAGER;
  my_allocator<<<1, 1>>>(nullptr, nullptr);
  printf("LONG LONG: %d\n", std::is_same_v<size_t, unsigned long long>);
}

__global__ void mpsc_kernel(
    hipc::MemoryBackend *backend,
    hipc::Allocator *alloc,
    hipc::mpsc_queue<int> *queue) {
  auto mem_mngr = HERMES_MEMORY_MANAGER;
  mem_mngr->AttachBackend(backend);
  mem_mngr->AttachAllocator(alloc);
  printf("%d %d",
         HERMES_MEMORY_MANAGER->GetDefaultAllocator() == alloc,
         HERMES_MEMORY_MANAGER->GetBackend(backend->header_->url_) == backend);
  queue->emplace(10);
}

void mpsc_test() {
  hshm::chararr shm_url = "test_serializers";
  hipc::allocator_id_t alloc_id(0, 1);
  auto mem_mngr = HERMES_MEMORY_MANAGER;
  mem_mngr->UnregisterAllocator(alloc_id);
  mem_mngr->UnregisterBackend(shm_url);
  auto *backend = mem_mngr->CreateBackend<hipc::CudaShmMmap>(
    MEGABYTES(100), shm_url, 0);
  hipc::Allocator *alloc = mem_mngr->CreateAllocator<hipc::ScalablePageAllocator>(
    shm_url, alloc_id, 0);
  auto queue = hipc::make_uptr<hipc::mpsc_queue<int>>(alloc, 256 * 256);
  printf("GetSize: %lu\n", queue->GetSize());
  mpsc_kernel<<<1, 1>>>(
    backend,
    alloc,
    queue.get());
  cudaDeviceSynchronize();
  printf("GetSize: %lu\n", queue->GetSize());
  int val, sum = 0;
  while (!queue->pop(val).IsNull()) {
    sum += val;
  }
  printf("SUM: %d\n", sum);
}

__global__ void atomic_kernel(hipc::atomic<hshm::min_u64> *x) {
  x->fetch_add(1);
}

void atomic_test() {
  hipc::atomic<hshm::min_u64> *x;
  cudaDeviceSynchronize();
  cudaSetDevice(0);
  size_t size = sizeof(hipc::atomic<hshm::min_u64>);
  cudaHostAlloc(&x, size, cudaHostAllocMapped);
  atomic_kernel<<<64, 64>>>(x);
  cudaDeviceSynchronize();
  printf("ATOMIC: %llu\n", x->load());
}


int main() {
  mpsc_test();
  // atomic_test();
}