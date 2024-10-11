//
// Created by llogan on 10/9/24.
//

#include <device_launch_parameters.h>
#include "hermes_shm/memory/backend/cuda_shm_mmap.h"

struct MyStruct {
  int x;
  float y;
};


__global__ void my_kernel(MyStruct* ptr) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 1) { // Only process the first block
    ptr->x = 100;
    ptr->y = 100;
    printf("Kernel: x=%d, y=%f\n", ptr->x, ptr->y);
  }
}

int main() {
  // Allocate memory on the host and device using UM
  size_t size = sizeof(MyStruct);

  // Create a MyStruct instance and copy it to both host and device memory
  MyStruct my_struct;
  my_struct.x = 10;
  my_struct.y = 3.14f;

  hshm::ipc::CudaShmMmap shm;
  shm.shm_init(size, "shmem_test", 0);
  memcpy(shm.data_, &my_struct, size);
  MyStruct* shm_struct = (MyStruct*)shm.data_;

  // Launch a CUDA kernel that accesses the shared memory
  int blockSize = 256;
  int numBlocks = 1;
  dim3 block(blockSize);
  dim3 grid(numBlocks);

  my_kernel<<<grid, block>>>(shm_struct);
  cudaDeviceSynchronize();
  // MyStruct new_struct = *shm_struct;

  // Free memory
  shm.shm_destroy();
}