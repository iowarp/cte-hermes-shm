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

#ifndef HERMES_INCLUDE_MEMORY_BACKEND_ROCM_SHM_MMAP_H
#define HERMES_INCLUDE_MEMORY_BACKEND_ROCM_SHM_MMAP_H

#include "memory_backend.h"
#include "hermes_shm/util/logging.h"
#include <string>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

#include <hermes_shm/util/errors.h>
#include <hermes_shm/constants/macros.h>
#include <hermes_shm/introspect/system_info.h>
#include "posix_shm_mmap.h"
#include <hip/hip_runtime.h>

namespace hshm::ipc {

class RocmShmMmap : public PosixShmMmap {
 public:
  /** Initialize shared memory */
  bool shm_init(const MemoryBackendId &backend_id,
                size_t size,
                const hshm::chararr &url,
                int device) {
    hipDeviceSynchronize();
    hipSetDevice(device);
    bool ret = PosixShmMmap::shm_init(backend_id, size, url);
    if (!ret) {
      return false;
    }
    header_->type_ = MemoryBackendType::kRocmShmMmap;
    return true;
  }

  /** Map shared memory */
  char* _Map(size_t size, off64_t off) override {
    char* ptr = _ShmMap(size, off);
    hipHostRegister(ptr, size, hipHostRegisterPortable);
    return ptr;
  }

  /** Detach shared memory */
  void _Detach() override {
    hipHostUnregister(header_);
    hipHostUnregister(data_);
    _ShmDetach();
  }
};

}  // namespace hshm::ipc

#endif  // HERMES_INCLUDE_MEMORY_BACKEND_ROCM_SHM_MMAP_H
