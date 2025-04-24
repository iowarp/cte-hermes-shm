//
// Created by llogan on 25/10/24.
//

#ifndef ROCM_MALLOC_H
#define ROCM_MALLOC_H

#include <hip/hip_runtime.h>

#include <string>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "hermes_shm/util/logging.h"
#include "memory_backend.h"
#include "posix_shm_mmap.h"

namespace hshm::ipc {

struct RocmMallocHeader : public MemoryBackendHeader {
  hipIpcMemHandle_t ipc_;
};

class RocmMalloc : public PosixShmMmap {
 public:
  CLS_CONST MemoryBackendType EnumType = MemoryBackendType::kRocmMalloc;

 public:
  /** Constructor */
  HSHM_CROSS_FUN
  RocmMalloc() = default;

  /** Destructor */
  ~RocmMalloc() {
    if (IsOwned()) {
      _Destroy();
    } else {
      _Detach();
    }
  }

  /** Initialize backend */
  bool shm_init(const MemoryBackendId &backend_id, size_t accel_data_size,
                const hshm::chararr &url, int device = 0,
                size_t md_size = KILOBYTES(4)) {
    bool ret = PosixShmMmap::shm_init(backend_id, md_size, url);
    if (!ret) {
      return false;
    }
    RocmMallocHeader *header = reinterpret_cast<RocmMallocHeader *>(header_);
    header->type_ = MemoryBackendType::kRocmMalloc;
    header->accel_data_size_ = accel_data_size;
    header->accel_id_ = device;
    accel_data_size_ = accel_data_size;
    accel_data_ = _Map(accel_data_size);
    accel_id_ = device;
    SetGpu();
    HIP_ERROR_CHECK(hipIpcGetMemHandle(&header->ipc_, (void *)accel_data_));
    return true;
  }

  /** Deserialize the backend */
  bool shm_deserialize(const hshm::chararr &url) {
    bool ret = PosixShmMmap::shm_deserialize(url);
    if (!ret) {
      return false;
    }
    RocmMallocHeader *header = reinterpret_cast<RocmMallocHeader *>(header_);
    accel_data_size_ = header_->accel_data_size_;
    accel_id_ = header->accel_id_;
    HIP_ERROR_CHECK(hipIpcOpenMemHandle((void **)&accel_data_, header->ipc_,
                                        hipIpcMemLazyEnablePeerAccess));
    SetGpu();
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
    HIP_ERROR_CHECK(hipMalloc(&ptr, size));
    return ptr;
  }

  /** Unmap shared memory */
  void _Detach() {}

  /** Destroy shared memory */
  void _Destroy() {
    if (!IsInitialized()) {
      return;
    }
    HIP_ERROR_CHECK(hipFree(header_));
    UnsetInitialized();
  }
};

}  // namespace hshm::ipc

#endif  // ROCM_MALLOC_H
