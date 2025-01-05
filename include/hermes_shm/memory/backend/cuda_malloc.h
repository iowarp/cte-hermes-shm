//
// Created by llogan on 25/10/24.
//

#ifndef CUDA_MALLOC_H
#define CUDA_MALLOC_H

#include <cuda_runtime.h>

#include <string>

#include "hermes_shm/constants/macros.h>
#include "hermes_shm/introspect/system_info.h>
#include "hermes_shm/util/errors.h>
#include "hermes_shm/util/logging.h"
#include "memory_backend.h"

namespace hshm::ipc {

class CudaMalloc : public MemoryBackend {
 private:
  size_t total_size_;

 public:
  /** Constructor */
  HSHM_CROSS_FUN
  CudaMalloc() = default;

  /** Destructor */
  ~CudaMalloc() override {
    if (IsOwned()) {
      _Destroy();
    } else {
      _Detach();
    }
  }

  /** Initialize backend */
  bool shm_init(const MemoryBackendId &backend_id, size_t size) {
    SetInitialized();
    Own();
    total_size_ = sizeof(MemoryBackendHeader) + size;
    char *ptr = _Map(total_size_);
    header_ = reinterpret_cast<MemoryBackendHeader *>(ptr);
    header_->type_ = MemoryBackendType::kCudaMalloc;
    header_->id_ = backend_id;
    header_->data_size_ = size;
    data_size_ = size;
    data_ = reinterpret_cast<char *>(header_ + 1);
    return true;
  }

  /** Deserialize the backend */
  bool shm_deserialize(const hshm::chararr &url) {
    (void)url;
    HERMES_THROW_ERROR(SHMEM_NOT_SUPPORTED);
  }

  /** Detach the mapped memory */
  void shm_detach() { _Detach(); }

  /** Destroy the mapped memory */
  void shm_destroy() { _Destroy(); }

 protected:
  /** Map shared memory */
  template <typename T = char>
  T *_Map(size_t size) {
    T *ptr;
    cudaMallocManaged(&ptr, size);
    return ptr;
  }

  /** Unmap shared memory */
  void _Detach() {}

  /** Destroy shared memory */
  void _Destroy() {
    if (!IsInitialized()) {
      return;
    }
    cudaFree(header_);
    UnsetInitialized();
  }
};

}  // namespace hshm::ipc

#endif  // CUDA_MALLOC_H
