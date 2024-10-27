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

#ifdef HERMES_ENABLE_CUDA
__global__ void Init() {
}
#endif

/** Create the root allocator */
HSHM_CROSS_FUN
MemoryManager::MemoryManager() {
  Init();
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
  MemoryBackend *backend = MemoryBackendFactory::shm_deserialize(type, url);
  RegisterBackend(backend);
  ScanBackends();
  backend->Disown();
  return backend;
#endif
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
    MemoryBackend *backend) {
  if (GetBackend(backend->GetId())) {
    HERMES_THROW_ERROR(MEMORY_BACKEND_REPEATED);
  }
  backends_[backend->GetId().id_] = backend;
  return backend;
}

#ifdef HERMES_ENABLE_CUDA
template<typename BackendT>
__global__ void AttachBackendKernel(BackendT *pack, BackendT *cpy) {
  HERMES_MEMORY_MANAGER;
  HERMES_THREAD_MODEL;
  HERMES_SYSTEM_INFO;
  new (cpy) BackendT(*pack);
  HERMES_MEMORY_MANAGER->RegisterBackend(cpy);
  HERMES_MEMORY_MANAGER->ScanBackends();
}

/** Check if a backend is cuda-compatible */
void AllocateCudaBackend(int dev, MemoryBackend *other) {
  cudaSetDevice(dev);
  switch(other->header_->type_) {
    case MemoryBackendType::kCudaMalloc: {
      CudaMalloc *pack, *cpy;
      cudaMallocManaged(&pack, sizeof(CudaMalloc));
      cudaMallocManaged(&cpy, sizeof(CudaMalloc));
      memcpy((char*)pack, (char*)other, sizeof(CudaMalloc));
      pack->UnsetScanned();
      pack->Disown();
      AttachBackendKernel<<<1, 1>>>(pack, cpy);
      cudaDeviceSynchronize();
      cudaFree(pack);
    }
    case MemoryBackendType::kCudaShmMmap: {
      CudaShmMmap *pack, *cpy;
      cudaMallocManaged(&pack, sizeof(CudaShmMmap));
      cudaMallocManaged(&cpy, sizeof(CudaShmMmap));
      memcpy((char*)pack, (char*)other, sizeof(CudaShmMmap));
      pack->UnsetScanned();
      pack->Disown();
      AttachBackendKernel<<<1, 1>>>(pack, cpy);
      cudaDeviceSynchronize();
      cudaFree(pack);
    }
    default: {
      break;
    }
  }
}
#endif

/**
 * Scans all attached backends for new memory allocators.
 * */
HSHM_CROSS_FUN
void MemoryManager::ScanBackends(bool find_allocs) {
#if defined(HERMES_ENABLE_CUDA) && !defined(__CUDA_ARCH__)
  int num_gpus = 0;
  cudaGetDeviceCount(&num_gpus);
#endif
  for (int i = 0; i < MAX_BACKENDS; ++i) {
    MemoryBackend *backend = backends_[i];
    if (backend == nullptr || backend->IsScanned()) {
      continue;
    }
    backend->SetScanned();
    if (find_allocs) {
      Allocator *alloc = AllocatorFactory::shm_deserialize(backend);
      if (!alloc) {
        continue;
      }
      RegisterAllocator(alloc, false);
    }
#if defined(HERMES_ENABLE_CUDA) && !defined(__CUDA_ARCH__)
    for (int dev = 0; dev < num_gpus; ++dev) {
      AllocateCudaBackend(dev, backend);
    }
#endif
  }
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
 * Registers an allocator. Used internally by ScanBackends, but may
 * also be used externally.
 * */
HSHM_CROSS_FUN
Allocator* MemoryManager::RegisterAllocator(Allocator *alloc, bool do_scan) {
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
  if (do_scan) {
    ScanBackends(false);
  }
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
