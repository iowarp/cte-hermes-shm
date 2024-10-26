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


#include "hermes_shm/memory/memory_manager.h"
#include "hermes_shm/memory/backend/memory_backend_factory.h"
#include "hermes_shm/memory/allocator/allocator_factory.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "hermes_shm/util/logging.h"
#include "hermes_shm/thread/thread_model_manager.h"
#include "hermes_shm/data_structures/ipc/unordered_map.h"

namespace hshm::ipc {

/** Create the root allocator */
HSHM_CROSS_FUN
MemoryManager::MemoryManager() {
#ifndef __CUDA_ARCH__
  Init();
#endif
}

/** Initialize memory manager */
HSHM_CROSS_FUN
void MemoryManager::Init() {
  // System info
  HERMES_SYSTEM_INFO->RefreshInfo();

  // Initialize tables
  memset(backends_, 0, sizeof(backends_));
  memset(allocators_, 0, sizeof(allocators_));

  // Root backend
  ArrayBackend *root_backend = (ArrayBackend*)root_backend_space_;
  Allocator::ConstructObj(*root_backend);
  root_backend->shm_init(MemoryBackendId::GetRoot(),
                         KILOBYTES(16), root_alloc_data_);
  root_backend->Own();
  root_backend_ = root_backend;

  // Root allocator
  root_allocator_id_.bits_.major_ = 3;
  root_allocator_id_.bits_.minor_ = 3;
  StackAllocator *root_alloc = (StackAllocator*)root_alloc_space_;
  Allocator::ConstructObj(*root_alloc);
  root_alloc->shm_init(
      root_allocator_id_, 0,
      root_backend_->data_,
      root_backend_->data_size_);
  root_alloc_ = root_alloc;
  default_allocator_ = root_alloc_;

  // Other allocators
  RegisterAllocator(root_alloc_);
  HERMES_THREAD_MODEL;
}

/** Get the root allocator */
HSHM_CROSS_FUN
Allocator* MemoryManager::GetRootAllocator() {
  return root_alloc_;
}

/** Default backend size */
HSHM_CROSS_FUN
size_t MemoryManager::GetDefaultBackendSize() {
#ifndef __CUDA_ARCH__
  return HERMES_SYSTEM_INFO->ram_size_;
#else
  // TODO(llogan)
#endif
}

/**
 * Attaches to an existing memory backend located at \a url url.
 * */
HSHM_CROSS_FUN
MemoryBackend* MemoryManager::AttachBackend(MemoryBackendType type,
                                            const hshm::chararr &url) {
#ifndef __CUDA_ARCH__
  auto backend = MemoryBackendFactory::shm_deserialize(type, url);
  RegisterBackend(backend->header_->id_, backend);
  ScanBackends();
  backend->Disown();
  return backend;
#endif
}

/**
 * Attaches to an existing memory backend.
 * */
HSHM_CROSS_FUN
MemoryBackend* MemoryManager::AttachBackend(MemoryBackend *other) {
  MemoryBackend *backend = MemoryBackendFactory::shm_attach(other);
  RegisterBackend(backend->header_->id_, backend);
  ScanBackends();
  backend->Disown();
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
HSHM_CROSS_FUN
MemoryBackend* MemoryManager::RegisterBackend(
    const MemoryBackendId &backend_id,
    MemoryBackend *backend) {
  if (GetBackend(backend_id)) {
    HERMES_THROW_ERROR(MEMORY_BACKEND_REPEATED);
  }
  backends_[backend_id.id_] = backend;
  return backend;
}

/**
 * Scans all attached backends for new memory allocators.
 * */
HSHM_CROSS_FUN
void MemoryManager::ScanBackends() {
#ifndef __CUDA_ARCH__
  for (int i = 0; i < MAX_BACKENDS; ++i) {
    auto alloc = AllocatorFactory::shm_deserialize(backends_[i]);
    RegisterAllocator(alloc);
  }
#endif
}


/**
 * Returns a pointer to a backend that has already been attached.
 * */
HSHM_CROSS_FUN
MemoryBackend* MemoryManager::GetBackend(const MemoryBackendId &backend_id) {
  return backends_[backend_id.id_];
}

/**
 * Unregister backend
 * */
HSHM_CROSS_FUN
void MemoryManager::UnregisterBackend(const MemoryBackendId &backend_id) {
  backends_[backend_id.id_] = nullptr;
}

/**
 * Destroy backend
 * */
HSHM_CROSS_FUN
void MemoryManager::DestroyBackend(const MemoryBackendId &backend_id) {
  auto backend = GetBackend(backend_id);
  backend->Own();
  UnregisterBackend(backend_id);
}

/**
 * Attaches an allocator that was previously allocated,
 * but was stored in shared memory. This is needed because
 * the virtual function table is not compatible with SHM.
 * */
HSHM_CROSS_FUN
void MemoryManager::AttachAllocator(Allocator *other) {
  Allocator *alloc = AllocatorFactory::shm_attach(other);
  RegisterAllocator(alloc);
}

/**
 * Registers an allocator. Used internally by ScanBackends, but may
 * also be used externally.
 * */
HSHM_CROSS_FUN
Allocator* MemoryManager::RegisterAllocator(Allocator *alloc) {
  if (alloc == nullptr) {
    return nullptr;
  }
  if (default_allocator_ == nullptr ||
      default_allocator_ == root_alloc_ ||
      default_allocator_->GetId() == alloc->GetId()) {
    default_allocator_ = alloc;
  }
  uint32_t idx = alloc->GetId().ToIndex();
  if (idx > MAX_ALLOCATORS) {
    HILOG(kError, "Allocator index out of range: {}", idx)
    HERMES_THROW_ERROR(TOO_MANY_ALLOCATORS);
  }
  allocators_[idx] = alloc;
  return alloc;
}

/**
 * Destroys an allocator
 * */
HSHM_CROSS_FUN
void MemoryManager::UnregisterAllocator(AllocatorId alloc_id) {
  if (alloc_id == default_allocator_->GetId()) {
    default_allocator_ = root_alloc_;
  }
  allocators_[alloc_id.ToIndex()] = nullptr;
}

/**
 * Locates an allocator of a particular id
 * */
HSHM_CROSS_FUN Allocator* MemoryManager::GetAllocator(AllocatorId alloc_id) {
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
