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


#include <hermes_shm/memory/memory_manager.h>
#include "hermes_shm/memory/backend/memory_backend_factory.h"
#include "hermes_shm/memory/allocator/allocator_factory.h"
#include <hermes_shm/introspect/system_info.h>
#include "hermes_shm/constants/data_structure_singleton_macros.h"

namespace hshm::ipc {

/** Create the root allocator */
HSHM_CROSS_FUN
MemoryManager::MemoryManager() {
  root_allocator_id_.bits_.major_ = 3;
  root_allocator_id_.bits_.minor_ = 3;
  root_backend_.shm_init(MEGABYTES(128));
  root_backend_.Own();
  root_allocator_ = GetRootAllocator();
  ((StackAllocator*)root_allocator_)->shm_init(
      root_allocator_id_, 0,
      root_backend_.data_,
      root_backend_.data_size_);
  default_allocator_ = root_allocator_;
  memset(allocators_, 0, sizeof(allocators_));
  RegisterAllocator(root_allocator_);
}

HSHM_CROSS_FUN
Allocator *MemoryManager::GetRootAllocator() {
  static StackAllocator root_allocator;
  return &root_allocator;
}

/** Default backend size */
HSHM_CROSS_FUN
size_t MemoryManager::GetDefaultBackendSize() {
  return HERMES_SYSTEM_INFO->ram_size_;
}

/**
 * Attaches to an existing memory backend located at \a url url.
 * */
HSHM_CROSS_FUN
MemoryBackend* MemoryManager::AttachBackend(MemoryBackendType type,
                                            const std::string &url) {
  auto backend_u = MemoryBackendFactory::shm_deserialize(type, url);
  auto backend = HERMES_MEMORY_MANAGER->RegisterBackend(url, backend_u);
  ScanBackends();
  backend->Disown();
  return backend;
}

/**
 * Scans all attached backends for new memory allocators.
 * */
HSHM_CROSS_FUN
void MemoryManager::ScanBackends() {
  for (auto &[url, backend] : backends_) {
    auto alloc = AllocatorFactory::shm_deserialize(backend.get());
    RegisterAllocator(alloc);
  }
}


/**
 * Returns a pointer to a backend that has already been attached.
 * */
HSHM_CROSS_FUN
MemoryBackend* MemoryManager::GetBackend(const std::string &url) {
  auto iter = backends_.find(url);
  if (iter == backends_.end()) {
    return nullptr;
  }
  return (*iter).second.get();
}

/**
 * Unregister backend
 * */
HSHM_CROSS_FUN
void MemoryManager::UnregisterBackend(const std::string &url) {
  backends_.erase(url);
}

/**
 * Destroy backend
 * */
HSHM_CROSS_FUN
void MemoryManager::DestroyBackend(const std::string &url) {
  auto backend = GetBackend(url);
  backend->Own();
  UnregisterBackend(url);
}

/**
 * Registers an allocator. Used internally by ScanBackends, but may
 * also be used externally.
 * */
HSHM_CROSS_FUN
Allocator* MemoryManager::RegisterAllocator(std::unique_ptr<Allocator> &alloc) {
  if (default_allocator_ == nullptr ||
      default_allocator_ == root_allocator_ ||
      default_allocator_->GetId() == alloc->GetId()) {
    default_allocator_ = alloc.get();
  }
  RegisterAllocator(alloc.get());
  auto idx = alloc->GetId().ToIndex();
  auto &alloc_made = allocators_made_[idx];
  alloc_made = std::move(alloc);
  return alloc_made.get();
}

/**
 * Registers an allocator. Used internally by ScanBackends, but may
 * also be used externally.
 * */
HSHM_CROSS_FUN
Allocator* MemoryManager::RegisterAllocator(Allocator *alloc) {
  uint32_t idx = alloc->GetId().ToIndex();
  if (idx > MAX_ALLOCATORS) {
    HILOG(kError, "Allocator index out of range: {}", idx)
    throw std::runtime_error("Too many allocators");
  }
  allocators_[idx] = alloc;
  return alloc;
}

/**
 * Destroys an allocator
 * */
HSHM_CROSS_FUN
void MemoryManager::UnregisterAllocator(allocator_id_t alloc_id) {
  if (alloc_id == default_allocator_->GetId()) {
    default_allocator_ = root_allocator_;
  }
  allocators_made_[alloc_id.ToIndex()] = nullptr;
  allocators_[alloc_id.ToIndex()] = nullptr;
}

/**
 * Locates an allocator of a particular id
 * */
HSHM_CROSS_FUN Allocator* MemoryManager::GetAllocator(allocator_id_t alloc_id) {
  return allocators_[alloc_id.ToIndex()];
}

/**
 * Gets the allocator used by default when no allocator is
 * used to construct an object.
 * */
HSHM_CROSS_FUN
Allocator* MemoryManager::GetDefaultAllocator() {
  return reinterpret_cast<Allocator*>(default_allocator_);
}

/**
 * Sets the allocator used by default when no allocator is
 * used to construct an object.
 * */
HSHM_CROSS_FUN
void MemoryManager::SetDefaultAllocator(Allocator *alloc) {
  default_allocator_ = alloc;
}

}  // namespace hshm::ipc
