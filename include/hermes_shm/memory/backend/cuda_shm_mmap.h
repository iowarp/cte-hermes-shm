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

#ifndef HERMES_INCLUDE_MEMORY_BACKEND_CUDA_SHM_MMAP_H
#define HERMES_INCLUDE_MEMORY_BACKEND_CUDA_SHM_MMAP_H

#include <cuda_runtime.h>
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

class CudaShmMmap : public PosixShmMmap {
 public:
  /** Initialize shared memory */
  bool shm_init(const MemoryBackendId& backend_id, size_t size,
                const hshm::chararr& url, int device) {
    cudaDeviceSynchronize();
    cudaSetDevice(device);
    bool ret = PosixShmMmap::shm_init(backend_id, size, url);
    if (!ret) {
      return false;
    }
    header_->type_ = MemoryBackendType::kCudaShmMmap;
    return true;
  }

  /** Map shared memory */
  char* _Map(size_t size, off64_t off) override {
    char* ptr = _ShmMap(size, off);
    cudaHostRegister(ptr, size, cudaHostRegisterPortable);
    return ptr;
  }

  /** Detach shared memory */
  void _Detach() override {
    cudaHostUnregister(header_);
    cudaHostUnregister(data_);
    _ShmDetach();
  }
};

}  // namespace hshm::ipc

#endif  // HERMES_INCLUDE_MEMORY_BACKEND_CUDA_SHM_MMAP_H
