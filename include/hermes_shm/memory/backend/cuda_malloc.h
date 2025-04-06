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

namespace hshm::ipc {

struct CudaMallocHeader : public MemoryBackendHeader {
  cudaIpcMemHandle_t ipc_;
};

class CudaMalloc : public MemoryBackend {
 public:
  CLS_CONST MemoryBackendType EnumType = MemoryBackendType::kCudaMalloc;

 private:
  size_t total_size_;

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
  bool shm_init(const MemoryBackendId &backend_id, size_t size,
                const hshm::chararr &url, int device = 0) {
    SetInitialized();
    Own();
    cudaDeviceSynchronize();
    cudaSetDevice(device);
    SystemInfo::DestroySharedMemory(url.c_str());
    if (!SystemInfo::CreateNewSharedMemory(
            fd_, url.c_str(), size + HSHM_SYSTEM_INFO->page_size_)) {
      char *err_buf = strerror(errno);
      HILOG(kError, "shm_open failed: {}", err_buf);
      return false;
    }
    url_ = url;
    header_ = (MemoryBackendHeader *)_ShmMap(HSHM_SYSTEM_INFO->page_size_, 0);
    CudaMallocHeader *header = reinterpret_cast<CudaMallocHeader *>(header_);
    header->type_ = MemoryBackendType::kCudaMalloc;
    header->id_ = backend_id;
    header->data_size_ = size;
    data_size_ = size;
    data_ = _Map(data_size_);
    HIP_ERROR_CHECK(cudaIpcGetMemHandle(&header->ipc_, (void *)data_));
    return true;
  }

  /** Deserialize the backend */
  bool shm_deserialize(const hshm::chararr &url) {
    SetInitialized();
    Disown();
    if (!SystemInfo::OpenSharedMemory(fd_, url.c_str())) {
      const char *err_buf = strerror(errno);
      HILOG(kError, "shm_open failed: {}", err_buf);
      return false;
    }
    header_ = (MemoryBackendHeader *)_ShmMap(HSHM_SYSTEM_INFO->page_size_, 0);
    CudaMallocHeader *header = reinterpret_cast<CudaMallocHeader *>(header_);
    data_size_ = header_->data_size_;
    HIP_ERROR_CHECK(cudaIpcOpenMemHandle((void **)&data_, header->ipc_,
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
