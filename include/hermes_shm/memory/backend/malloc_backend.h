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

#ifndef HSHM_INCLUDE_HSHM_MEMORY_BACKEND_MALLOC_H
#define HSHM_INCLUDE_HSHM_MEMORY_BACKEND_MALLOC_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "memory_backend.h"

namespace hshm::ipc {

class MallocBackend : public MemoryBackend {
 public:
  CLS_CONST MemoryBackendType EnumType = MemoryBackendType::kMallocBackend;

 private:
  size_t total_size_;

 public:
  HSHM_CROSS_FUN
  MallocBackend() = default;

  ~MallocBackend() {}

  HSHM_CROSS_FUN
  bool shm_init(const MemoryBackendId &backend_id, size_t size) {
    SetInitialized();
    Own();
    total_size_ = sizeof(MemoryBackendHeader) + size;
    char *ptr = (char *)malloc(total_size_);
    header_ = reinterpret_cast<MemoryBackendHeader *>(ptr);
    header_->type_ = MemoryBackendType::kMallocBackend;
    header_->id_ = backend_id;
    header_->data_size_ = size;
    data_size_ = size;
    data_ = (char *)(header_ + 1);
    return true;
  }

  bool shm_deserialize(const hshm::chararr &url) {
    (void)url;
    HSHM_THROW_ERROR(SHMEM_NOT_SUPPORTED);
    return false;
  }

  void shm_detach() { _Detach(); }

  void shm_destroy() { _Destroy(); }

 protected:
  void _Detach() { free(header_); }

  void _Destroy() { free(header_); }
};

}  // namespace hshm::ipc

#endif  // HSHM_INCLUDE_HSHM_MEMORY_BACKEND_MALLOC_H
