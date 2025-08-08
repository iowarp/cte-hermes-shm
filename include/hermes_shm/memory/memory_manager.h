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

#ifndef HSHM_MEMORY_MEMORY_MANAGER_H_
#define HSHM_MEMORY_MEMORY_MANAGER_H_

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/memory/allocator/allocator_factory.h"
#include "hermes_shm/memory/backend/memory_backend_factory.h"
#include "hermes_shm/memory/memory_manager_.h"
#include "hermes_shm/util/logging.h"
#include "memory.h"

namespace hshm::ipc {

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
/** Copy the memory manager */
static HSHM_GPU_KERNEL void CopyMemoryManager(MemoryManager **mm) {
  memcpy(HSHM_MEMORY_MANAGER, *mm, sizeof(MemoryManager));
}

/** Get the memory manager GPU pointer */
static HSHM_GPU_KERNEL void GetMemoryManagerGpuKern(MemoryManager **mm) {
  printf("Memory Manager: %p\n", HSHM_MEMORY_MANAGER);
  *mm = HSHM_MEMORY_MANAGER;
}

/** Register a backend on GPU */
static HSHM_GPU_KERNEL void RegisterBackendGpuKern(MemoryBackendId backend_id,
                                                   char *region, size_t size) {
  // printf("HSHM: Registering backend on a GPU: %u\n", backend_id.id_);
#if HSHM_IS_GPU
  HSHM_MEMORY_MANAGER;
  HSHM_THREAD_MODEL;
  HSHM_SYSTEM_INFO;
  auto alloc = HSHM_ROOT_ALLOC;
  auto full_ptr =
      alloc->template NewObj<hipc::ArrayBackend>(HSHM_DEFAULT_MEM_CTX);
  auto backend = full_ptr.ptr_;
  backend->local_hdr_.id_ = backend_id;
  if (!backend->shm_init(backend_id, size, region)) {
    HSHM_THROW_ERROR(MEMORY_BACKEND_CREATE_FAILED);
  }
  HSHM_MEMORY_MANAGER->RegisterBackend(backend);
  backend->Own();
  HSHM_MEMORY_MANAGER->ScanBackends();
#endif
  printf("HSHM (%p): Registered backend on a GPU: %u \n", HSHM_MEMORY_MANAGER,
         backend_id.id_);
}

/** Scan backends on GPU */
static HSHM_GPU_KERNEL void ScanBackendGpuKern() {
  // printf("HSHM: Scanning backends on GPU\n");
  HSHM_MEMORY_MANAGER->ScanBackends();
}

/** Create an allocator on GPU */
template <typename AllocT, typename... Args>
static HSHM_GPU_KERNEL void CreateAllocatorGpuKern(MemoryBackendId backend_id,
                                                   AllocatorId alloc_id,
                                                   size_t custom_header_size,
                                                   Args... args) {
  printf("HSHM: CreateAllocatorGpuKern: %d.%d\n", alloc_id.bits_.major_,
         alloc_id.bits_.minor_);
  auto *backend = HSHM_MEMORY_MANAGER->GetBackend(backend_id);
  if (!backend) {
    printf("HSHM: CreateAllocatorGpuKern: backend is null\n");
    return;
  }
  HSHM_MEMORY_MANAGER->CreateAllocator<AllocT>(
      backend_id, alloc_id, custom_header_size, std::forward<Args>(args)...);
  auto *alloc = HSHM_MEMORY_MANAGER->GetAllocator<AllocT>(alloc_id);
  printf("HSHM: Created allocator on GPU: (%p) %d.%d\n", alloc,
         alloc_id.bits_.major_, alloc_id.bits_.minor_);
}

/** Mark a GPU as having an allocator */
static HSHM_GPU_KERNEL void SetBackendHasAllocGpuKern(
    MemoryBackendId backend_id) {
  // printf("HSHM: SetBackendHasAllocGpuKern\n");
  MemoryBackend *backend = HSHM_MEMORY_MANAGER->GetBackend(backend_id);
  if (!backend) {
    return;
  }
  backend->SetHasAlloc();
}
#endif  // if defined(HSHM_ENABLE_CUDA) || defined(HSHM_ENABLE_ROCM)

/** Initialize memory manager */
template <int>
HSHM_CROSS_FUN void MemoryManager::Init() {
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
  RegisterAllocatorNoScan(root_alloc_);

// Get the memory manager GPU pointer
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
#if HSHM_IS_HOST
  int ngpu = GpuApi::GetDeviceCount();
  for (int gpu_id = 0; gpu_id < ngpu; ++gpu_id) {
    GpuApi::SetDevice(gpu_id);
    gpu_ptrs_[gpu_id] =
        GpuApi::Malloc<MemoryManager *>(sizeof(MemoryManager *));
    GetMemoryManagerGpuKern<<<1, 1>>>(gpu_ptrs_[gpu_id]);
    GpuApi::Synchronize();
  }
#endif
#endif
}

/**
 * Registers an allocator. Used internally by ScanBackends, but may
 * also be used externally.
 * */
template <int>
HSHM_CROSS_FUN Allocator *MemoryManager::RegisterAllocator(Allocator *alloc) {
  if (alloc == nullptr) {
    return nullptr;
  }
  RegisterAllocatorNoScan(alloc);
  ScanBackends();
  return alloc;
}

/**
 * Create a memory backend. Memory backends are divided into slots.
 * Each slot corresponds directly with a single allocator.
 * There can be multiple slots per-backend, enabling multiple allocation
 * policies over a single memory region.
 * */
template <typename BackendT, typename... Args>
MemoryBackend *MemoryManager::CreateBackend(const MemoryBackendId &backend_id,
                                            size_t size, Args &&...args) {
  auto backend = MemoryBackendFactory::shm_init<BackendT>(
      backend_id, size, std::forward<Args>(args)...);
  RegisterBackend(backend);
  backend->Own();
  if (backend->IsCopyGpu()) {
    CopyBackendGpu(backend_id);
  }
  return backend;
}

/**
 * Attaches to an existing memory backend located at \a url url.
 * */
template <int>
HSHM_CROSS_FUN MemoryBackend *MemoryManager::AttachBackend(
    MemoryBackendType type, const hshm::chararr &url) {
#if HSHM_IS_HOST
  MemoryBackend *backend = MemoryBackendFactory::shm_deserialize(type, url);
  RegisterBackend(backend);
  if (backend->IsCopyGpu()) {
    CopyBackendGpu(backend->GetId());
  }
  ScanBackends();
  backend->Disown();
  return backend;
#else
  return nullptr;
#endif
}

/**
 * Copies all backends to GPU
 */
template <int>
void MemoryManager::CopyAllBackendsToGpu() {
  for (int i = 0; i < HSHM_MAX_BACKENDS; ++i) {
    MemoryBackend *backend = backends_[i];
    if (backend == nullptr || !backend->IsCopyGpu()) {
      continue;
    }
    CopyBackendGpu(backend->GetId());
  }
  ScanBackendsAllGpu();
}

/**
 * Destroys a backned
 * */
template <int>
HSHM_CROSS_FUN void MemoryManager::DestroyBackend(
    const MemoryBackendId &backend_id) {
  auto backend = UnregisterBackend(backend_id);
  if (backend == nullptr) {
    return;
  }
  FullPtr<MemoryBackend> ptr(backend);
  backend->Own();
  auto alloc = GetAllocator<HSHM_ROOT_ALLOC_T>(ptr.shm_.alloc_id_);
  alloc->DelObj(HSHM_DEFAULT_MEM_CTX, ptr);
}

/**
 * Scans all attached backends for new memory allocators.
 * */
template <int>
HSHM_CROSS_FUN void MemoryManager::ScanBackends() {
  for (int i = 0; i < HSHM_MAX_BACKENDS; ++i) {
    MemoryBackend *backend = backends_[i];
    if (backend == nullptr) {
      continue;
    }
    if (!backend->IsHasAlloc()) {
      continue;
    }
    if (backend->IsScanned()) {
      continue;
    }
    auto *alloc = AllocatorFactory::shm_deserialize(backend);
    if (!alloc) {
      continue;
    }
    // printf("HSHM: Discovered allocator on %s: %u\n", kCurrentDevice,
    //        backend->GetId().id_);
    backend->SetScanned();
    RegisterAllocatorNoScan(alloc);
  }
#if HSHM_IS_HOST
  ScanBackendsAllGpu();
#endif
}

/**
 * Scans backends on all GPUs
 */
template <int>
void MemoryManager::ScanBackendsAllGpu() {
  int ngpu = GpuApi::GetDeviceCount();
  for (int gpu_id = 0; gpu_id < ngpu; ++gpu_id) {
    ScanBackendsGpu(gpu_id);
  }
}

/**
 * Scans backends on the GPU
 */
template <int nothing>
void MemoryManager::ScanBackendsGpu(int gpu_id) {
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  GpuApi::SetDevice(gpu_id);
  ScanBackendGpuKern<<<1, 1>>>();
  GpuApi::Synchronize();
#endif
}

/**
 * Create and register a memory allocator for a particular backend.
 * */
template <typename AllocT, typename... Args>
HSHM_INLINE_CROSS_FUN AllocT *MemoryManager::CreateAllocator(
    const MemoryBackendId &backend_id, const AllocatorId &alloc_id,
    size_t custom_header_size, Args &&...args) {
  MemoryBackend *backend = GetBackend(backend_id);
  if (alloc_id.IsNull()) {
    HELOG(kFatal, "Allocator cannot be created with a NIL ID");
  }
  if (backend == nullptr) {
    return nullptr;
  }
  backend->SetHasAlloc();
  if (backend->IsMirrorGpu()) {
    backend->SetHasGpuAlloc();
  }
  if (backend->IsHasGpuAlloc()) {
    SetBackendHasAllocGpu(backend->GetId());
  }
  AllocT *alloc = AllocatorFactory::shm_init<AllocT>(
      alloc_id, custom_header_size, backend, std::forward<Args>(args)...);
  RegisterAllocator(alloc);
  return GetAllocator<AllocT>(alloc_id);
}

/**
 * Create + register allocator on GPU.
 * */
template <typename AllocT, typename... Args>
void MemoryManager::CreateAllocatorGpu(int gpu_id,
                                       const MemoryBackendId &backend_id,
                                       const AllocatorId &alloc_id,
                                       size_t custom_header_size,
                                       Args &&...args) {
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  if (alloc_id.IsNull()) {
    HELOG(kFatal, "Allocator cannot be created with a NIL ID");
  }
  GpuApi::SetDevice(gpu_id);
  HSHM_MEMORY_MANAGER->CreateAllocator<NullAllocator>(backend_id, alloc_id,
                                                      custom_header_size);
  CreateAllocatorGpuKern<AllocT><<<1, 1>>>(
      backend_id, alloc_id, custom_header_size, std::forward<Args>(args)...);
  GpuApi::Synchronize();
  MemoryBackend *backend = GetBackend(backend_id);
  if (backend) {
    backend->SetHasGpuAlloc();
  }
#endif
}

/**
 * Destroys an allocator
 * */
template <typename AllocT>
HSHM_CROSS_FUN void MemoryManager::DestroyAllocator(
    const AllocatorId &alloc_id) {
  auto dead_alloc = UnregisterAllocator(alloc_id);
  if (dead_alloc == nullptr) {
    return;
  }
  FullPtr<AllocT> ptr((AllocT *)dead_alloc);
  auto alloc = GetAllocator<HSHM_ROOT_ALLOC_T>(ptr.shm_.alloc_id_);
  alloc->template DelObj<AllocT>(HSHM_DEFAULT_MEM_CTX, ptr);
}

/** Default backend size */
template <int>
HSHM_CROSS_FUN size_t MemoryManager::GetDefaultBackendSize() {
#if HSHM_IS_HOST
  return HSHM_SYSTEM_INFO->ram_size_;
#else
  // TODO(llogan)
  return 0;
#endif
}

/** Copy and existing backend to the GPU */
template <int>
void MemoryManager::CopyBackendGpu(const MemoryBackendId &backend_id) {
  MemoryBackend *backend = GetBackend(backend_id);
  if (!backend) {
    return;
  }
  GpuApi::SetDevice(backend->accel_id_);
  CreateBackendGpu(backend->accel_id_, backend_id, backend->accel_data_,
                   backend->accel_data_size_);
  if (backend->IsHasGpuAlloc()) {
    SetBackendHasAllocGpu(backend_id);
  }
}

/** Create an array backend on the GPU */
template <int>
void MemoryManager::CreateBackendGpu(int gpu_id,
                                     const MemoryBackendId &backend_id,
                                     char *accel_data, size_t accel_data_size) {
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  GpuApi::SetDevice(gpu_id);
  RegisterBackendGpuKern<<<1, 1>>>(backend_id, accel_data, accel_data_size);
  GpuApi::Synchronize();
#endif
}

/** Set a backend as having an allocator */
template <int>
void MemoryManager::SetBackendHasAllocGpu(const MemoryBackendId &backend_id) {
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  MemoryBackend *backend = GetBackend(backend_id);
  if (!backend) {
    return;
  }
  GpuApi::SetDevice(backend->accel_id_);
  SetBackendHasAllocGpuKern<<<1, 1>>>(backend_id);
  GpuApi::Synchronize();
#endif
}

/**
 * Full Pointer Constructors
 */
template <typename T, typename PointerT>
HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT>::FullPtr(const PointerT &shm)
    : shm_(shm) {
  ptr_ = HSHM_MEMORY_MANAGER->Convert<T, PointerT>(shm);
}

template <typename T, typename PointerT>
HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT>::FullPtr(const T *ptr)
    : ptr_(const_cast<T *>(ptr)) {
  shm_ = HSHM_MEMORY_MANAGER->Convert<T, PointerT>(ptr_);
}

template <typename T, typename PointerT>
HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT>::FullPtr(hipc::Allocator *alloc,
                                                    const T *ptr)
    : ptr_(const_cast<T *>(ptr)) {
  shm_ = alloc->Convert<T, PointerT>(ptr);
}

template <typename T, typename PointerT>
HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT>::FullPtr(hipc::Allocator *alloc,
                                                    const OffsetPointer &shm) {
  ptr_ = alloc->Convert<T, OffsetPointer>(shm);
  shm_ = PointerT(alloc->GetId(), shm);
}

}  // namespace hshm::ipc

#endif  // HSHM_MEMORY_MEMORY_MANAGER_H_
