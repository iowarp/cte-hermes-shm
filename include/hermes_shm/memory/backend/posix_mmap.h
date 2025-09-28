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

#ifndef HSHM_INCLUDE_MEMORY_BACKEND_POSIX_MMAP_H
#define HSHM_INCLUDE_MEMORY_BACKEND_POSIX_MMAP_H

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include "hermes_shm/constants/macros.h"
#if HSHM_ENABLE_PROCFS_SYSINFO
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "memory_backend.h"

namespace hshm::ipc {

class PosixMmap : public MemoryBackend {
 public:
  CLS_CONST MemoryBackendType EnumType = MemoryBackendType::kPosixMmap;

 private:
  size_t total_size_;

 public:
  /** Constructor */
  HSHM_CROSS_FUN
  PosixMmap() = default;

  /** Destructor */
  ~PosixMmap() {
    if (IsOwned()) {
      _Destroy();
    } else {
      _Detach();
    }
  }

  /** Initialize backend */
  bool shm_init(const MemoryBackendId &backend_id, size_t size) {
    SetInitialized();
    Own();
    total_size_ = sizeof(MemoryBackendHeader) + size;
    char *ptr = _Map(total_size_);
    header_ = reinterpret_cast<MemoryBackendHeader *>(ptr);
    header_->type_ = MemoryBackendType::kPosixMmap;
    header_->id_ = backend_id;
    header_->data_size_ = size;
    data_size_ = size;
    data_ = reinterpret_cast<char *>(header_ + 1);
    return true;
  }

  /** Deserialize the backend */
  bool shm_deserialize(const hshm::chararr &url) {
    (void)url;
    HSHM_THROW_ERROR(SHMEM_NOT_SUPPORTED);
    return false;
  }

  /** Detach the mapped memory */
  void shm_detach() { _Detach(); }

  /** Destroy the mapped memory */
  void shm_destroy() { _Destroy(); }

 protected:
  /** Map shared memory */
  template <typename T = char>
  T *_Map(size_t size) {
    T *ptr = reinterpret_cast<T *>(
        SystemInfo::MapPrivateMemory(MemoryAlignment::AlignToPageSize(size)));
    if (!ptr) {
      HSHM_THROW_ERROR(SHMEM_CREATE_FAILED);
    }
    return ptr;
  }

  /** Unmap shared memory */
  void _Detach() {
    if (!IsInitialized()) {
      return;
    }
    SystemInfo::UnmapMemory(reinterpret_cast<void *>(header_), total_size_);
    UnsetInitialized();
  }

  /** Destroy shared memory */
  void _Destroy() {
    if (!IsInitialized()) {
      return;
    }
    _Detach();
    UnsetInitialized();
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_INCLUDE_MEMORY_BACKEND_POSIX_MMAP_H
