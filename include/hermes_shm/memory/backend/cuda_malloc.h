//
// Created by llogan on 25/10/24.
//

#ifndef CUDA_MALLOC_H
#define CUDA_MALLOC_H

#include <cuda_runtime.h>

#include <string>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "hermes_shm/util/logging.h"
#include "memory_backend.h"
#include "posix_shm_mmap.h"

namespace hshm::ipc {

struct CudaMallocHeader : public MemoryBackendHeader {
  cudaIpcMemHandle_t ipc_;
};

class CudaMalloc : public PosixShmMmap {
 public:
  CLS_CONST MemoryBackendType EnumType = MemoryBackendType::kCudaMalloc;

 public:
  /** Constructor */
  HSHM_CROSS_FUN
  CudaMalloc() = default;

  /** Destructor */
  ~CudaMalloc() {
    if (IsOwned()) {
      _Destroy();
    } else {
      _Detach();
    }
  }

  /** Initialize backend */
  bool shm_init(const MemoryBackendId &backend_id, size_t accel_data_size,
                const hshm::chararr &url, int device = 0,
                size_t md_size = MEGABYTES(1)) {
    bool ret = PosixShmMmap::shm_init(backend_id, md_size, url);
    if (!ret) {
      return false;
    }
    CudaMallocHeader *header = reinterpret_cast<CudaMallocHeader *>(header_);
    header->type_ = MemoryBackendType::kCudaMalloc;
    header->accel_data_size_ = accel_data_size;
    accel_data_size_ = accel_data_size;
    accel_data_ = _Map(accel_data_size);
    CUDA_ERROR_CHECK(cudaIpcGetMemHandle(&header->ipc_, (void *)accel_data_));
    return true;
  }

  /** Deserialize the backend */
  bool shm_deserialize(const hshm::chararr &url) {
    bool ret = PosixShmMmap::shm_deserialize(url);
    if (!ret) {
      return false;
    }
    CudaMallocHeader *header = reinterpret_cast<CudaMallocHeader *>(header_);
    accel_data_size_ = header_->accel_data_size_;
    CUDA_ERROR_CHECK(cudaIpcOpenMemHandle((void **)&accel_data_, header->ipc_,
                                          cudaIpcMemLazyEnablePeerAccess));
    return true;
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
    cudaMalloc(&ptr, size);
    return ptr;
  }

  /** Unmap shared memory */
  void _Detach() {}

  /** Destroy shared memory */
  void _Destroy() {
    if (!IsInitialized()) {
      return;
    }
    UnsetInitialized();
  }
};

}  // namespace hshm::ipc

#endif  // CUDA_MALLOC_H
