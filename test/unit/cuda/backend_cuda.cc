//
// Created by llogan on 10/9/24.
//

#include <stdio.h>
#include "hermes_shm/memory/backend/cuda_shm_mmap.h"
#include "hermes_shm/constants/macros.h"
#include "hermes_shm/types/argpack.h"
#include "hermes_shm/util/singleton/_easy_lockfree_singleton.h"
#include <cassert>

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

__global__ void my_kernel(MyStruct* ptr) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  MyStruct quest;
  ptr->x = quest.DoSomething();
  ptr->y = hshm::PassArgPack::Call(
      hshm::make_argpack(0, 1, 2),
      [](int x, int y, int z) {
        return x + y + z;
      });
  *hshm::EasyLockfreeSingleton<int>::GetInstance() = 25;
  ptr->x = *hshm::EasyLockfreeSingleton<int>::GetInstance();
  // MyStruct::GetInstance() = 16;
}

int main() {
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

  my_kernel<<<grid, block>>>(shm_struct);
  cudaDeviceSynchronize();

  // Verify correctness
  MyStruct new_struct = *shm_struct;
  assert(new_struct.x == 25);
  assert(new_struct.y == 3);
  // assert(MyStruct::GetInstance() == 222);

  // Free memory
  shm.shm_destroy();
}