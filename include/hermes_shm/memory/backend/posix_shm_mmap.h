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

#ifndef HERMES_INCLUDE_MEMORY_BACKEND_POSIX_SHM_MMAP_H
#define HERMES_INCLUDE_MEMORY_BACKEND_POSIX_SHM_MMAP_H

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "hermes_shm/util/logging.h"
#include "memory_backend.h"

namespace hshm::ipc {

class PosixShmMmap : public MemoryBackend, public UrlMemoryBackend {
 protected:
  File fd_;
  hshm::chararr url_;

 public:
  /** Constructor */
  HSHM_CROSS_FUN
  PosixShmMmap() {}

  /** Destructor */
  HSHM_CROSS_FUN
  ~PosixShmMmap() override {
#ifdef HSHM_IS_HOST
    if (IsOwned()) {
      _Destroy();
    } else {
      _Detach();
    }
#endif
  }

  /** Initialize backend */
  bool shm_init(const MemoryBackendId &backend_id, size_t size,
                const hshm::chararr &url) {
    SetInitialized();
    Own();
    SystemInfo::DestroySharedMemory(url.c_str());
    if (!SystemInfo::CreateNewSharedMemory(
            fd_, url.c_str(), size + HERMES_SYSTEM_INFO->page_size_)) {
      char *err_buf = strerror(errno);
      HILOG(kError, "shm_open failed: {}", err_buf);
      return false;
    }
    url_ = url;
    header_ = (MemoryBackendHeader *)_Map(HERMES_SYSTEM_INFO->page_size_, 0);
    header_->type_ = MemoryBackendType::kPosixShmMmap;
    header_->id_ = backend_id;
    header_->data_size_ = size;
    data_size_ = size;
    data_ = _Map(size, HERMES_SYSTEM_INFO->page_size_);
    return true;
  }

  /** Deserialize the backend */
  bool shm_deserialize(const hshm::chararr &url) override {
    SetInitialized();
    Disown();
    if (!SystemInfo::OpenSharedMemory(fd_, url.c_str())) {
      const char *err_buf = strerror(errno);
      HILOG(kError, "shm_open failed: {}", err_buf);
      return false;
    }
    header_ = (MemoryBackendHeader *)_Map(HERMES_SYSTEM_INFO->page_size_, 0);
    data_size_ = header_->data_size_;
    data_ = _Map(data_size_, HERMES_SYSTEM_INFO->page_size_);
    return true;
  }

  /** Detach the mapped memory */
  void shm_detach() override { _Detach(); }

  /** Destroy the mapped memory */
  void shm_destroy() override { _Destroy(); }

 protected:
  /** Map shared memory */
  virtual char *_Map(size_t size, i64 off) { return _ShmMap(size, off); };

  /** Map shared memory */
  char *_ShmMap(size_t size, i64 off) {
    char *ptr =
        reinterpret_cast<char *>(SystemInfo::MapSharedMemory(fd_, size, off));
    if (!ptr) {
      HERMES_THROW_ERROR(SHMEM_CREATE_FAILED);
    }
    return ptr;
  }

  /** Unmap shared memory (virtual) */
  virtual void _Detach() { _ShmDetach(); }

  /** Unmap shared memory */
  void _ShmDetach() {
    if (!IsInitialized()) {
      return;
    }
    SystemInfo::UnmapMemory(reinterpret_cast<void *>(header_),
                            HERMES_SYSTEM_INFO->page_size_);
    SystemInfo::UnmapMemory(data_, data_size_);
    SystemInfo::CloseSharedMemory(fd_);
    UnsetInitialized();
  }

  /** Destroy shared memory */
  void _Destroy() {
    if (!IsInitialized()) {
      return;
    }
    _Detach();
    SystemInfo::DestroySharedMemory(url_.c_str());
    UnsetInitialized();
  }
};

}  // namespace hshm::ipc

#endif  // HERMES_INCLUDE_MEMORY_BACKEND_POSIX_SHM_MMAP_H
