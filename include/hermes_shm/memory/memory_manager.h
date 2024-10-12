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
#include <hermes_shm/constants/data_structure_singleton_macros.h>
#include "memory_manager_.h"

namespace hipc = hshm::ipc;

namespace hshm::ipc {

/**
 * Create a memory backend. Memory backends are divided into slots.
 * Each slot corresponds directly with a single allocator.
 * There can be multiple slots per-backend, enabling multiple allocation
 * policies over a single memory region.
 * */
template<typename BackendT, typename ...Args>
HSHM_CROSS_FUN
MemoryBackend* MemoryManager::CreateBackend(size_t size,
                             const std::string &url,
                             Args&& ...args) {
  auto backend_u = MemoryBackendFactory::shm_init<BackendT>(
    size, url, std::forward<Args>(args)...);
  auto backend = RegisterBackend(url, backend_u);
  backend->Own();
  return backend;
}

/**
 * Register a unique memory backend. Throws an exception if the backend
 * already exists. This is because unregistering a backend can cause
 * ramifications across allocators.
 *
 * @param url the backend's unique identifier
 * @param backend the backend to register
 * */
HSHM_INLINE_CROSS_FUN
MemoryBackend* MemoryManager::RegisterBackend(
    const std::string &url,
    std::unique_ptr<MemoryBackend> &backend) {
  auto ptr = backend.get();
  if (GetBackend(url)) {
    throw MEMORY_BACKEND_REPEATED.format();
  }
  backends_.emplace(url, std::move(backend));
  return ptr;
}

/**
 * Create and register a memory allocator for a particular backend.
 * */
template<typename AllocT, typename ...Args>
HSHM_CROSS_FUN
Allocator* MemoryManager::CreateAllocator(const std::string &url,
                                          allocator_id_t alloc_id,
                                          size_t custom_header_size,
                                          Args&& ...args) {
  auto backend = GetBackend(url);
  if (alloc_id.IsNull()) {
    HELOG(kFatal, "Allocator cannot be created with a NIL ID");
  }
  auto alloc = AllocatorFactory::shm_init<AllocT>(
      alloc_id, custom_header_size, backend, std::forward<Args>(args)...);
  RegisterAllocator(alloc);
  return GetAllocator(alloc_id);
}

}  // namespace hshm::ipc

#endif  // HERMES_MEMORY_MEMORY_MANAGER_H_
