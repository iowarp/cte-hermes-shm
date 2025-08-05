//
// Created by llogan on 25/10/24.
//

#ifndef GPU_MALLOC_H
#define GPU_MALLOC_H

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <string>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "hermes_shm/util/gpu_api.h"
#include "hermes_shm/util/logging.h"
#include "memory_backend.h"
#include "posix_shm_mmap.h"

namespace hshm::ipc {

struct GpuMallocHeader : public MemoryBackendHeader {
  GpuIpcMemHandle ipc_;
};

class GpuMalloc : public PosixShmMmap {
 public:
  CLS_CONST MemoryBackendType EnumType = MemoryBackendType::kGpuMalloc;

 public:
  /** Constructor */
  HSHM_CROSS_FUN
  GpuMalloc() = default;

  /** Destructor */
  ~GpuMalloc() {
    if (IsOwned()) {
      _Destroy();
    } else {
      _Detach();
    }
  }

  /** Initialize backend */
  bool shm_init(const MemoryBackendId &backend_id, size_t accel_data_size,
                const hshm::chararr &url, int gpu_id = 0,
                size_t md_size = MEGABYTES(1)) {
    bool ret = PosixShmMmap::shm_init(backend_id, md_size, url);
    if (!ret) {
      return false;
    }
    SetCopyGpu();
    GpuMallocHeader *header = reinterpret_cast<GpuMallocHeader *>(header_);
    header->type_ = MemoryBackendType::kGpuMalloc;
    header->accel_data_size_ = accel_data_size;
    header->accel_id_ = gpu_id;
    accel_data_size_ = accel_data_size;
    accel_data_ = _Map(accel_data_size);
    accel_id_ = header_->accel_id_;
    GpuApi::GetIpcMemHandle(header->ipc_, (void *)accel_data_);
    return true;
  }

  /** Deserialize the backend */
  bool shm_deserialize(const hshm::chararr &url) {
    bool ret = PosixShmMmap::shm_deserialize(url);
    if (!ret) {
      return false;
    }
    SetCopyGpu();
    GpuMallocHeader *header = reinterpret_cast<GpuMallocHeader *>(header_);
    accel_data_size_ = header_->accel_data_size_;
    accel_id_ = header_->accel_id_;
    GpuApi::OpenIpcMemHandle(header->ipc_, &accel_data_);
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
    return GpuApi::Malloc<T>(size);
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

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#endif  // GPU_MALLOC_H
