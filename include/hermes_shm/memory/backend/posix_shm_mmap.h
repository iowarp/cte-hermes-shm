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

#ifndef HSHM_INCLUDE_MEMORY_BACKEND_POSIX_SHM_MMAP_H
#define HSHM_INCLUDE_MEMORY_BACKEND_POSIX_SHM_MMAP_H

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
 public:
  CLS_CONST MemoryBackendType EnumType = MemoryBackendType::kPosixShmMmap;

 protected:
  File fd_;
  hshm::chararr url_;
  CLS_CONST int hdr_size_ = KILOBYTES(16);

 public:
  /** Constructor */
  HSHM_CROSS_FUN
  PosixShmMmap() {}

  /** Destructor */
  HSHM_CROSS_FUN
  ~PosixShmMmap() {
#if HSHM_IS_HOST
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
    std::string url_s = url.str();
    SystemInfo::DestroySharedMemory(url_s);
    if (!SystemInfo::CreateNewSharedMemory(fd_, url_s, size + hdr_size_)) {
      char *err_buf = strerror(errno);
      HILOG(kError, "shm_open failed: {}", err_buf);
      return false;
    }
    url_ = url;
    header_ = (MemoryBackendHeader *)_ShmMap(hdr_size_, 0);
    new (header_) MemoryBackendHeader();
    header_->type_ = MemoryBackendType::kPosixShmMmap;
    header_->id_ = backend_id;
    header_->data_size_ = size;
    data_size_ = size;
    data_ = _ShmMap(size, hdr_size_);
    return true;
  }

  /** Deserialize the backend */
  bool shm_deserialize(const hshm::chararr &url) {
    SetInitialized();
    Disown();
    std::string url_s = url.str();
    if (!SystemInfo::OpenSharedMemory(fd_, url_s)) {
      const char *err_buf = strerror(errno);
      HILOG(kError, "shm_open failed: {}", err_buf);
      return false;
    }
    header_ = (MemoryBackendHeader *)_ShmMap(hdr_size_, 0);
    data_size_ = header_->data_size_;
    data_ = _ShmMap(data_size_, hdr_size_);
    return true;
  }

  /** Detach the mapped memory */
  void shm_detach() { _Detach(); }

  /** Destroy the mapped memory */
  void shm_destroy() { _Destroy(); }

 protected:
  /** Map shared memory */
  char *_ShmMap(size_t size, i64 off) {
    char *ptr =
        reinterpret_cast<char *>(SystemInfo::MapSharedMemory(fd_, size, off));
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
    SystemInfo::UnmapMemory(reinterpret_cast<void *>(header_),
                            HSHM_SYSTEM_INFO->page_size_);
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

#endif  // HSHM_INCLUDE_MEMORY_BACKEND_POSIX_SHM_MMAP_H
