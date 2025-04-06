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

#ifndef HSHM_INCLUDE_MEMORY_BACKEND_ROCM_SHM_MMAP_H
#define HSHM_INCLUDE_MEMORY_BACKEND_ROCM_SHM_MMAP_H

#include <fcntl.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "hermes_shm/util/logging.h"
#include "memory_backend.h"
#include "posix_shm_mmap.h"

namespace hshm::ipc {

class RocmShmMmap : public PosixShmMmap {
 public:
  CLS_CONST MemoryBackendType EnumType = MemoryBackendType::kRocmShmMmap;

 public:
  /** Constructor */
  HSHM_CROSS_FUN
  RocmShmMmap() {}

  /** Destructor */
  HSHM_CROSS_FUN
  ~RocmShmMmap() {
#ifdef HSHM_IS_HOST
    if (IsOwned()) {
      _Destroy();
    } else {
      _Detach();
    }
#endif
  }

  /** Initialize shared memory */
  bool shm_init(const MemoryBackendId& backend_id, size_t size,
                const hshm::chararr& url, int device) {
    HIP_ERROR_CHECK(hipDeviceSynchronize());
    HIP_ERROR_CHECK(hipSetDevice(device));
    bool ret = PosixShmMmap::shm_init(backend_id, size, url);
    Register(header_, HSHM_SYSTEM_INFO->page_size_);
    Register(data_, size);
    if (!ret) {
      return false;
    }
    header_->type_ = MemoryBackendType::kRocmShmMmap;
    return true;
  }

  /** SHM deserialize */
  bool shm_deserialize(const hshm::chararr& url) {
    bool ret = PosixShmMmap::shm_deserialize(url);
    Register(header_, HSHM_SYSTEM_INFO->page_size_);
    Register(data_, data_size_);
    return ret;
  }

  /** Map shared memory */
  template <typename T>
  void Register(T* ptr, size_t size) {
    HIP_ERROR_CHECK(hipHostRegister((void*)ptr, size, hipHostRegisterPortable));
  }

  /** Detach shared memory */
  void _Detach() {
    HIP_ERROR_CHECK(hipHostUnregister(header_));
    HIP_ERROR_CHECK(hipHostUnregister(data_));
    PosixShmMmap::_Detach();
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_INCLUDE_MEMORY_BACKEND_ROCM_SHM_MMAP_H
