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

#ifndef HERMES_MEMORY_MEMORY_MANAGER_H_
#define HERMES_MEMORY_MEMORY_MANAGER_H_

#include "hermes_shm/memory/backend/memory_backend_factory.h"
#include "hermes_shm/memory/allocator/allocator_factory.h"
#include "hermes_shm/memory/memory_manager_.h"
#include "hermes_shm/constants/macros.h"
#include "hermes_shm/util/logging.h"
#include "memory.h"

namespace hshm::ipc {

/**
 * Create a memory backend. Memory backends are divided into slots.
 * Each slot corresponds directly with a single allocator.
 * There can be multiple slots per-backend, enabling multiple allocation
 * policies over a single memory region.
 * */
template<typename BackendT, typename ...Args>
MemoryBackend* MemoryManager::CreateBackend(
    const MemoryBackendId &backend_id,
    size_t size,
    Args&& ...args) {
  auto backend = MemoryBackendFactory::shm_init<BackendT>(
      backend_id, size, std::forward<Args>(args)...);
  RegisterBackend(backend);
  backend->Own();
  return backend;
}

/**
 * Create and register a memory allocator for a particular backend.
 * */
template<typename AllocT, typename ...Args>
AllocT* MemoryManager::CreateAllocator(const MemoryBackendId &backend_id,
                                          const AllocatorId &alloc_id,
                                          size_t custom_header_size,
                                          Args&& ...args) {
  MemoryBackend *backend = GetBackend(backend_id);
  if (alloc_id.IsNull()) {
    HELOG(kFatal, "Allocator cannot be created with a NIL ID");
  }
  AllocT *alloc = AllocatorFactory::shm_init<AllocT>(
      alloc_id, custom_header_size, backend, std::forward<Args>(args)...);
  RegisterAllocator(alloc);
  return GetAllocator<AllocT>(alloc_id);
}

template <typename T, typename PointerT>
LPointer<T, PointerT>::LPointer(const PointerT &shm) : shm_(shm) {
  ptr_ = HERMES_MEMORY_MANAGER->Convert<T, PointerT>(shm);
}

template <typename T, typename PointerT>
LPointer<T, PointerT>::LPointer(T *ptr) : ptr_(ptr) {
  shm_ = HERMES_MEMORY_MANAGER->Convert<T, PointerT>(ptr);
}

}  // namespace hshm::ipc

#endif  // HERMES_MEMORY_MEMORY_MANAGER_H_
