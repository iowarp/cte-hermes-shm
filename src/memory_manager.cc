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

#define HSHM_COMPILING_DLL
#define __HSHM_IS_COMPILING__

#include "hermes_shm/memory/memory_manager.h"

#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/memory/allocator/allocator_factory.h"
#include "hermes_shm/memory/backend/memory_backend_factory.h"
#include "hermes_shm/thread/thread_model_manager.h"
#include "hermes_shm/util/errors.h"
#include "hermes_shm/util/logging.h"

namespace hshm::ipc {

/** Create the root allocator */
HSHM_CROSS_FUN
MemoryManager::MemoryManager() { Init(); }

/** Initialize memory manager */
HSHM_CROSS_FUN
void MemoryManager::Init() {
  // System info
  HSHM_SYSTEM_INFO->RefreshInfo();

  // Initialize tables
  memset(backends_, 0, sizeof(backends_));
  memset(allocators_, 0, sizeof(allocators_));

  // Root backend
  ArrayBackend *root_backend = (ArrayBackend *)root_backend_space_;
  Allocator::ConstructObj(*root_backend);
  root_backend->shm_init(MemoryBackendId::GetRoot(), sizeof(root_alloc_data_),
                         root_alloc_data_);
  root_backend->Own();
  root_backend_ = root_backend;

  // Root allocator
  root_alloc_id_.bits_.major_ = 0;
  root_alloc_id_.bits_.minor_ = 0;
  StackAllocator *root_alloc = (StackAllocator *)root_alloc_space_;
  Allocator::ConstructObj(*root_alloc);
  root_alloc->shm_init(root_alloc_id_, 0, *root_backend_);
  root_alloc_ = root_alloc;
  default_allocator_ = root_alloc_;

  // Other allocators
  RegisterAllocator(root_alloc_);
}

/** Default backend size */
HSHM_CROSS_FUN
size_t MemoryManager::GetDefaultBackendSize() {
#ifdef HSHM_IS_HOST
  return HSHM_SYSTEM_INFO->ram_size_;
#else
  // TODO(llogan)
  return 0;
#endif
}

/**
 * Attaches to an existing memory backend located at \a url url.
 * */
HSHM_CROSS_FUN
MemoryBackend *MemoryManager::AttachBackend(MemoryBackendType type,
                                            const hshm::chararr &url) {
#ifdef HSHM_IS_HOST
  MemoryBackend *backend = MemoryBackendFactory::shm_deserialize(type, url);
  RegisterBackend(backend);
  ScanBackends();
  backend->Disown();
  return backend;
#else
  return nullptr;
#endif
}

/**
 * Destroys a backned
 * */
HSHM_CROSS_FUN void MemoryManager::DestroyBackend(
    const MemoryBackendId &backend_id) {
  auto backend = UnregisterBackend(backend_id);
  if (backend == nullptr) {
    return;
  }
  FullPtr<MemoryBackend> ptr(backend);
  backend->Own();
  auto alloc = GetAllocator<HSHM_ROOT_ALLOC_T>(ptr.shm_.alloc_id_);
  alloc->DelObjLocal(HSHM_DEFAULT_MEM_CTX, ptr);
}

#if defined(HSHM_ENABLE_CUDA) || defined(HSHM_ENABLE_ROCM)
HSHM_GPU_KERNEL void RegisterBackendGpuKern(const MemoryBackendId &backend_id,
                                            char *region, size_t size) {
  HSHM_MEMORY_MANAGER;
  HSHM_THREAD_MODEL;
  HSHM_SYSTEM_INFO;
  auto alloc = HSHM_ROOT_ALLOC;
  auto backend =
      alloc->template NewObj<hipc::ArrayBackend>(HSHM_DEFAULT_MEM_CTX);
  if (!backend->shm_init(backend_id, size, region)) {
    HSHM_THROW_ERROR(MEMORY_BACKEND_CREATE_FAILED);
  }
  HSHM_MEMORY_MANAGER->RegisterBackend(backend);
  backend->Own();
}

HSHM_GPU_KERNEL void ScanBackendGpuKern(bool find_allocs) {
  HSHM_MEMORY_MANAGER->ScanBackends(find_allocs);
}
#endif

/** Copy and existing backend to the GPU */
void MemoryManager::CopyBackendGpu(int gpu_id,
                                   const MemoryBackendId &backend_id) {
  GpuApi::SetDevice(gpu_id);
  MemoryBackend *backend = GetBackend(backend_id);
  if (!backend) {
    return;
  }
  CreateBackendGpu(gpu_id, backend_id, backend->accel_data_,
                   backend->accel_data_size_);
}

/** Create an array backend on the GPU */
void MemoryManager::CreateBackendGpu(int gpu_id,
                                     const MemoryBackendId &backend_id,
                                     char *accel_data, size_t accel_data_size) {
#if defined(HSHM_ENABLE_CUDA) || defined(HSHM_ENABLE_ROCM)
  GpuApi::SetDevice(gpu_id);
  RegisterBackendGpuKern<<<1, 1>>>(backend_id, accel_data, accel_data_size);
  GpuApi::Synchronize();
#endif
}

/**
 * Scans all attached backends for new memory allocators.
 * */
HSHM_CROSS_FUN
void MemoryManager::ScanBackends(bool find_allocs) {
  for (int i = 0; i < MAX_BACKENDS; ++i) {
    MemoryBackend *backend = backends_[i];
    if (backend == nullptr || backend->IsScanned()) {
      continue;
    }
    backend->SetScanned();
    if (find_allocs) {
      auto *alloc = AllocatorFactory::shm_deserialize(backend);
      if (!alloc) {
        continue;
      }
      RegisterAllocator(alloc, false);
    }
  }

#ifdef HSHM_IS_HOST
  int ngpu = GpuApi::GetDeviceCount();
  for (int gpu_id = 0; gpu_id < ngpu; ++gpu_id) {
    ScanBackendsGpu(gpu_id, find_allocs);
  }
#endif
}

/**
 * Scans backends on the GPU
 */
HSHM_HOST_FUN
void MemoryManager::ScanBackendsGpu(int gpu_id, bool find_allocs) {
#if defined(HSHM_ENABLE_CUDA) || defined(HSHM_ENABLE_ROCM)
  GpuApi::SetDevice(gpu_id);
  ScanBackendGpuKern<<<1, 1>>>(find_allocs);
  GpuApi::Synchronize();
#endif
}

/**
 * Registers an allocator. Used internally by ScanBackends, but may
 * also be used externally.
 * */
HSHM_CROSS_FUN
Allocator *MemoryManager::RegisterAllocator(Allocator *alloc, bool do_scan) {
  if (alloc == nullptr) {
    return nullptr;
  }
  if (do_scan) {
    if (default_allocator_ == nullptr || default_allocator_ == root_alloc_ ||
        default_allocator_->GetId() == alloc->GetId()) {
      default_allocator_ = alloc;
    }
  }
  uint32_t idx = alloc->GetId().ToIndex();
  if (idx > MAX_ALLOCATORS) {
    HILOG(kError, "Allocator index out of range: {}", idx);
    HSHM_THROW_ERROR(TOO_MANY_ALLOCATORS);
  }
  allocators_[idx] = alloc;
  if (do_scan) {
    ScanBackends(false);
  }
  return alloc;
}

HSHM_DEFINE_GLOBAL_CROSS_PTR_VAR_CC(hshm::ipc::MemoryManager,
                                    hshmMemoryManager);

}  // namespace hshm::ipc

// TODO(llogan): Fix. A hack for HIP compiler to function
// I would love to spend more time figuring out why ROCm
// Fails to compile without this, but whatever.
#include "hermes_shm/introspect/system_info.cc"
