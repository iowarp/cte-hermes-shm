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


#ifndef HERMES_MEMORY_BACKEND_MEMORY_BACKEND_FACTORY_H_
#define HERMES_MEMORY_BACKEND_MEMORY_BACKEND_FACTORY_H_

#include "memory_backend.h"
#include "posix_mmap.h"
#include "posix_shm_mmap.h"
#include "malloc_backend.h"
#include "array_backend.h"
#include "hermes_shm/memory/memory_manager_.h"
#ifdef HERMES_ENABLE_CUDA
#include "cuda_shm_mmap.h"
#include "cuda_malloc.h"
#endif

namespace hshm::ipc {

#define HSHM_CREATE_BACKEND(T) \
  if constexpr(std::is_same_v<T, BackendT>) { \
    auto alloc = HERMES_MEMORY_MANAGER->GetRootAllocator();\
    auto backend = alloc->NewObj<T>(hshm::ThreadId::GetNull());\
    if (!backend->shm_init(backend_id, size, std::forward<Args>(args)...)) {\
      HERMES_THROW_ERROR(MEMORY_BACKEND_CREATE_FAILED);\
    }\
    return backend;\
  }

#define HSHM_DESERIALIZE_BACKEND(T)\
  case MemoryBackendType::k##T: {\
    auto alloc = HERMES_MEMORY_MANAGER->GetRootAllocator();\
    auto backend = alloc->NewObj<T>(hshm::ThreadId::GetNull());\
    if (!backend->shm_deserialize(url)) {\
      HERMES_THROW_ERROR(MEMORY_BACKEND_NOT_FOUND);\
    }\
    return backend;\
  }

class MemoryBackendFactory {
 public:
  /** Initialize a new backend */
  template<typename BackendT, typename ...Args>
  static MemoryBackend* shm_init(
    const MemoryBackendId &backend_id, size_t size, Args ...args) {
    // HSHM_CREATE_BACKEND(PosixShmMmap)
    if constexpr(std::is_same_v<PosixShmMmap, BackendT>) {
      auto alloc = HERMES_MEMORY_MANAGER->GetRootAllocator();
      auto backend = alloc->NewObj<PosixShmMmap>(hshm::ThreadId::GetNull());
      if (!backend->shm_init(backend_id, size, std::forward<Args>(args)...)) {
        HERMES_THROW_ERROR(MEMORY_BACKEND_CREATE_FAILED);
      }
      return backend;
    }
#ifdef HERMES_ENABLE_CUDA
    HSHM_CREATE_BACKEND(CudaShmMmap)
    HSHM_CREATE_BACKEND(CudaMalloc)
#endif
    HSHM_CREATE_BACKEND(PosixMmap)
    HSHM_CREATE_BACKEND(MallocBackend)
    HSHM_CREATE_BACKEND(ArrayBackend)

    // Error handling
    HERMES_THROW_ERROR(MEMORY_BACKEND_NOT_FOUND);
  }

  /** Deserialize an existing backend */
  static MemoryBackend* shm_deserialize(
    MemoryBackendType type, const hshm::chararr &url) {
    switch (type) {
      HSHM_DESERIALIZE_BACKEND(PosixShmMmap)
#ifdef HERMES_ENABLE_CUDA
      HSHM_DESERIALIZE_BACKEND(CudaShmMmap)
      HSHM_DESERIALIZE_BACKEND(CudaMalloc)
#endif
      HSHM_DESERIALIZE_BACKEND(PosixMmap)
      HSHM_DESERIALIZE_BACKEND(MallocBackend)
      HSHM_DESERIALIZE_BACKEND(ArrayBackend)

      // Default
      default: return nullptr;
    }
  }

  /** Deserialize an existing backend */
  HSHM_CROSS_FUN
  static MemoryBackend* shm_attach(MemoryBackend *backend) {
    switch (backend->header_->type_) {
      // Posix Mmap
      case MemoryBackendType::kPosixMmap: {
        return HERMES_MEMORY_MANAGER->GetRootAllocator()->
          NewObjLocal<PosixMmap>(ThreadId::GetNull(),
                                 *(PosixMmap*)backend).ptr_;
      }

        // Malloc
      case MemoryBackendType::kMallocBackend: {
        return HERMES_MEMORY_MANAGER->GetRootAllocator()->
        NewObjLocal<MallocBackend>(ThreadId::GetNull(),
                                   *(MallocBackend*)backend).ptr_;
      }

        // Array
      case MemoryBackendType::kArrayBackend: {
        return HERMES_MEMORY_MANAGER->GetRootAllocator()->
          NewObjLocal<ArrayBackend>(ThreadId::GetNull(),
                                    *(ArrayBackend*)backend).ptr_;
      }

#ifdef HERMES_ENABLE_CUDA
        // Cuda Malloc
      case MemoryBackendType::kCudaMalloc: {
        return HERMES_MEMORY_MANAGER->GetRootAllocator()->
          NewObjLocal<CudaMalloc>(ThreadId::GetNull(),
                                  *(CudaMalloc*)backend).ptr_;
      }

        // Cuda Shm Mmap
      case MemoryBackendType::kCudaShmMmap: {
        return HERMES_MEMORY_MANAGER->GetRootAllocator()->
          NewObjLocal<CudaShmMmap>(ThreadId::GetNull(),
                                   *(CudaShmMmap*)backend).ptr_;
      }
#endif

        // Default
      default: {
        HERMES_THROW_ERROR(MEMORY_BACKEND_NOT_FOUND);
      }
    }

    return nullptr;
  }
};

}  // namespace hshm::ipc

#endif  // HERMES_MEMORY_BACKEND_MEMORY_BACKEND_FACTORY_H_
