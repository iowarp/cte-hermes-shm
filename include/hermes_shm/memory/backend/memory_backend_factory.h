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

#ifndef HSHM_MEMORY_BACKEND_MEMORY_BACKEND_FACTORY_H_
#define HSHM_MEMORY_BACKEND_MEMORY_BACKEND_FACTORY_H_

#include "array_backend.h"
#include "hermes_shm/memory/allocator/allocator_factory.h"
#include "hermes_shm/memory/memory_manager_.h"
#include "malloc_backend.h"
#include "memory_backend.h"
#include "posix_mmap.h"
#include "posix_shm_mmap.h"
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
#include "gpu_malloc.h"
#include "gpu_shm_mmap.h"
#endif

namespace hshm::ipc {

#define HSHM_CREATE_BACKEND(T)                                               \
  if constexpr (std::is_same_v<T, BackendT>) {                               \
    auto alloc = HSHM_ROOT_ALLOC;                                            \
    auto full_ptr = alloc->template NewObj<T>(HSHM_DEFAULT_MEM_CTX);         \
    auto backend = full_ptr.ptr_;                                            \
    if (!backend->shm_init(backend_id, size, std::forward<Args>(args)...)) { \
      HSHM_THROW_ERROR(MEMORY_BACKEND_CREATE_FAILED);                        \
    }                                                                        \
    return backend;                                                          \
  }

#define HSHM_DESERIALIZE_BACKEND(T)                                 \
  case MemoryBackendType::k##T: {                                   \
    auto alloc = HSHM_ROOT_ALLOC;                                   \
    auto full_ptr = alloc->template NewObj<T>(HSHM_DEFAULT_MEM_CTX);\
    auto backend = full_ptr.ptr_;                                   \
    if (!backend->shm_deserialize(url)) {                           \
      HSHM_THROW_ERROR(MEMORY_BACKEND_NOT_FOUND);                   \
    }                                                               \
    return backend;                                                 \
  }

class MemoryBackendFactory {
 public:
  /** Initialize a new backend */
  template <typename BackendT, typename... Args>
  static MemoryBackend *shm_init(const MemoryBackendId &backend_id, size_t size,
                                 Args... args) {
    HSHM_CREATE_BACKEND(PosixShmMmap)
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
    HSHM_CREATE_BACKEND(GpuShmMmap)
    HSHM_CREATE_BACKEND(GpuMalloc)
#endif

    HSHM_CREATE_BACKEND(PosixMmap)
    HSHM_CREATE_BACKEND(MallocBackend)
    HSHM_CREATE_BACKEND(ArrayBackend)

    // Error handling
    HSHM_THROW_ERROR(MEMORY_BACKEND_NOT_FOUND);
  }

  /** Deserialize an existing backend */
  static MemoryBackend *shm_deserialize(MemoryBackendType type,
                                        const hshm::chararr &url) {
    switch (type) {
      HSHM_DESERIALIZE_BACKEND(PosixShmMmap)
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
      HSHM_DESERIALIZE_BACKEND(GpuShmMmap)
      HSHM_DESERIALIZE_BACKEND(GpuMalloc)
#endif
      HSHM_DESERIALIZE_BACKEND(PosixMmap)
      HSHM_DESERIALIZE_BACKEND(MallocBackend)
      HSHM_DESERIALIZE_BACKEND(ArrayBackend)

      // Default
      default:
        return nullptr;
    }
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_MEMORY_BACKEND_MEMORY_BACKEND_FACTORY_H_
