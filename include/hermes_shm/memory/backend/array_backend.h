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

#ifndef HERMES_INCLUDE_HERMES_MEMORY_BACKEND_ARRAY_BACKEND_H_
#define HERMES_INCLUDE_HERMES_MEMORY_BACKEND_ARRAY_BACKEND_H_

#include "memory_backend.h"
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

namespace hshm::ipc {

class ArrayBackend : public MemoryBackend {
 public:
  HSHM_CROSS_FUN
  ArrayBackend() = default;

  HSHM_CROSS_FUN
  ~ArrayBackend() override {}

  HSHM_CROSS_FUN
  bool shm_init(size_t size, char *region) {
    if (size < sizeof(MemoryBackendHeader)) {
      HERMES_THROW_ERROR(SHMEM_CREATE_FAILED);
    }
    SetInitialized();
    Own();
    header_ = reinterpret_cast<MemoryBackendHeader *>(region);
    header_->data_size_ = size - sizeof(MemoryBackendHeader);
    data_size_ = header_->data_size_;
    data_ = region + sizeof(MemoryBackendHeader);
    return true;
  }

  HSHM_CROSS_FUN
  bool shm_deserialize(const hshm::chararr &url) override {
    (void) url;
    HERMES_THROW_ERROR(SHMEM_NOT_SUPPORTED);
  }

  HSHM_CROSS_FUN
  void shm_detach() override {}

  HSHM_CROSS_FUN
  void shm_destroy() override {}
};

}  // namespace hshm::ipc

#endif  // HERMES_INCLUDE_HERMES_MEMORY_BACKEND_ARRAY_BACKEND_H_
