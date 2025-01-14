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
  /** Initialize shared memory */
  bool shm_init(const MemoryBackendId& backend_id, size_t size,
                const hshm::chararr& url, int device) {
    HIP_ERROR_CHECK(hipDeviceSynchronize());
    HIP_ERROR_CHECK(hipSetDevice(device));
    bool ret = PosixShmMmap::shm_init(backend_id, size, url);
    if (!ret) {
      return false;
    }
    header_->type_ = MemoryBackendType::kRocmShmMmap;
    return true;
  }

  /** Map shared memory */
  char* _Map(size_t size, i64 off) override {
    char* ptr = _ShmMap(size, off);
    HIP_ERROR_CHECK(hipHostRegister(ptr, size, hipHostRegisterPortable));
    return ptr;
  }

  /** Detach shared memory */
  void _Detach() override {
    HIP_ERROR_CHECK(hipHostUnregister(header_));
    HIP_ERROR_CHECK(hipHostUnregister(data_));
    _ShmDetach();
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_INCLUDE_MEMORY_BACKEND_ROCM_SHM_MMAP_H
