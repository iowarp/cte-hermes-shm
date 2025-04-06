//
// Created by llogan on 25/10/24.
//

#ifndef ROCM_MALLOC_H
#define ROCM_MALLOC_H

#include <hip/hip_runtime.h>

#include <string>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "hermes_shm/util/logging.h"
#include "memory_backend.h"

namespace hshm::ipc {

struct RocmMallocHeader : public MemoryBackendHeader {
  hipIpcMemHandle_t ipc_;
};

class RocmMalloc : public MemoryBackend {
 public:
  CLS_CONST MemoryBackendType EnumType = MemoryBackendType::kRocmMalloc;

 private:
  File fd_;
  hshm::chararr url_;

 public:
  /** Constructor */
  HSHM_CROSS_FUN
  RocmMalloc() = default;

  /** Destructor */
  ~RocmMalloc() {
    if (IsOwned()) {
      _Destroy();
    } else {
      _Detach();
    }
  }

  /**
   * Initialize backend
   * @param backend_id The backend id
   * @param size The size of the shared memory
   * @param url The url of the shared memory
   * @param unused Unused parameter used for template factories
   */
  bool shm_init(const MemoryBackendId &backend_id, size_t size,
                const hshm::chararr &url, int unused = 0) {
    SetInitialized();
    Own();
    SystemInfo::DestroySharedMemory(url.c_str());
    if (!SystemInfo::CreateNewSharedMemory(
            fd_, url.c_str(), size + HSHM_SYSTEM_INFO->page_size_)) {
      char *err_buf = strerror(errno);
      HILOG(kError, "shm_open failed: {}", err_buf);
      return false;
    }
    url_ = url;
    header_ = (MemoryBackendHeader *)_ShmMap(HSHM_SYSTEM_INFO->page_size_, 0);
    RocmMallocHeader *header = reinterpret_cast<RocmMallocHeader *>(header_);
    header->type_ = MemoryBackendType::kRocmMalloc;
    header->id_ = backend_id;
    header->data_size_ = size;
    data_size_ = size;
    data_ = _Map(data_size_);
    HIP_ERROR_CHECK(hipIpcGetMemHandle(&header->ipc_, (void *)data_));
    return true;
  }

  /** Deserialize the backend */
  bool shm_deserialize(const hshm::chararr &url) {
    SetInitialized();
    Disown();
    if (!SystemInfo::OpenSharedMemory(fd_, url.c_str())) {
      const char *err_buf = strerror(errno);
      HILOG(kError, "shm_open failed: {}", err_buf);
      return false;
    }
    header_ = (MemoryBackendHeader *)_ShmMap(HSHM_SYSTEM_INFO->page_size_, 0);
    RocmMallocHeader *header = reinterpret_cast<RocmMallocHeader *>(header_);
    data_size_ = header_->data_size_;
    HIP_ERROR_CHECK(hipIpcOpenMemHandle((void **)&data_, header->ipc_,
                                        hipIpcMemLazyEnablePeerAccess));
    return true;
  }

  /** Detach the mapped memory */
  void shm_detach() { _Detach(); }

  /** Destroy the mapped memory */
  void shm_destroy() { _Destroy(); }

 protected:
  /** Map shared memory */
  template <typename T = char>
  T *_Map(size_t size) {
    T *ptr;
    HIP_ERROR_CHECK(hipMallocManaged(&ptr, size));
    return ptr;
  }

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
  void _Detach() {}

  /** Destroy shared memory */
  void _Destroy() {
    if (!IsInitialized()) {
      return;
    }
    HIP_ERROR_CHECK(hipFree(header_));
    UnsetInitialized();
  }
};

}  // namespace hshm::ipc

#endif  // ROCM_MALLOC_H
