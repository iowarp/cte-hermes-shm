/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Hermes. The full Hermes copyright notice, including  *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory. If you do not  *
 * have access to the file, you may request a copy from help@hdfgroup.org.   *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef HSHM_INCLUDE_MEMORY_BACKEND_GPU_SHM_MMAP_H
#define HSHM_INCLUDE_MEMORY_BACKEND_GPU_SHM_MMAP_H

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "hermes_shm/util/gpu_api.h"
#include "hermes_shm/util/logging.h"
#include "memory_backend.h"
#include "posix_shm_mmap.h"

namespace hshm::ipc {

class GpuShmMmap : public PosixShmMmap {
 public:
  CLS_CONST MemoryBackendType EnumType = MemoryBackendType::kGpuShmMmap;

 public:
  /** Constructor */
  HSHM_CROSS_FUN
  GpuShmMmap() {}

  /** Destructor */
  HSHM_CROSS_FUN
  ~GpuShmMmap() {
#if HSHM_IS_HOST
    if (IsOwned()) {
      _Destroy();
    } else {
      _Detach();
    }
#endif
  }

  /** Initialize shared memory */
  bool shm_init(const MemoryBackendId& backend_id, size_t size,
                const hshm::chararr& url, int gpu_id = 0) {
    bool ret = PosixShmMmap::shm_init(backend_id, size, url);
    if (!ret) {
      return false;
    }
    SetCopyGpu();
    SetMirrorGpu();
    Register(header_, HSHM_SYSTEM_INFO->page_size_);
    Register(data_, size);
    header_->accel_data_size_ = data_size_;
    header_->accel_id_ = gpu_id;
    accel_data_ = data_;
    accel_data_size_ = data_size_;
    accel_id_ = header_->accel_id_;
    header_->type_ = MemoryBackendType::kGpuShmMmap;
    return true;
  }

  /** SHM deserialize */
  bool shm_deserialize(const hshm::chararr& url) {
    SetCopyGpu();
    SetMirrorGpu();
    bool ret = PosixShmMmap::shm_deserialize(url);
    Register(header_, HSHM_SYSTEM_INFO->page_size_);
    Register(data_, data_size_);
    accel_data_ = data_;
    accel_data_size_ = data_size_;
    accel_id_ = header_->accel_id_;
    return ret;
  }

  /** Map shared memory */
  template <typename T>
  void Register(T* ptr, size_t size) {
    GpuApi::RegisterHostMemory(ptr, size);
  }

  /** Detach shared memory */
  void _Detach() {
    GpuApi::UnregisterHostMemory(header_);
    GpuApi::UnregisterHostMemory(data_);
    PosixShmMmap::_Detach();
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#endif  // HSHM_INCLUDE_MEMORY_BACKEND_GPU_SHM_MMAP_H
