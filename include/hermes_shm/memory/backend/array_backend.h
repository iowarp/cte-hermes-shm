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

#ifndef HSHM_INCLUDE_HSHM_MEMORY_BACKEND_ARRAY_BACKEND_H_
#define HSHM_INCLUDE_HSHM_MEMORY_BACKEND_ARRAY_BACKEND_H_

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "memory_backend.h"

namespace hshm::ipc {

class ArrayBackend : public MemoryBackend {
 public:
  CLS_CONST MemoryBackendType EnumType = MemoryBackendType::kArrayBackend;
  MemoryBackendHeader local_hdr_;

 public:
  HSHM_CROSS_FUN
  ArrayBackend() = default;

  ~ArrayBackend() {}

  HSHM_CROSS_FUN
  bool shm_init(const MemoryBackendId &backend_id, size_t size, char *region) {
    if (size < sizeof(MemoryBackendHeader)) {
      HSHM_THROW_ERROR(SHMEM_CREATE_FAILED);
    }
    SetInitialized();
    Own();
    header_ = &local_hdr_;
    local_hdr_.type_ = MemoryBackendType::kArrayBackend;
    local_hdr_.id_ = backend_id;
    local_hdr_.data_size_ = size;
    data_size_ = local_hdr_.data_size_;
    data_ = region;
    return true;
  }

  bool shm_deserialize(const hshm::chararr &url) {
    (void)url;
    HSHM_THROW_ERROR(SHMEM_NOT_SUPPORTED);
    return false;
  }

  void shm_detach() {}

  void shm_destroy() {}
};

}  // namespace hshm::ipc

#endif  // HSHM_INCLUDE_HSHM_MEMORY_BACKEND_ARRAY_BACKEND_H_
