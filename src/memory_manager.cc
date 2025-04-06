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
template <typename BackendT>
HSHM_GPU_KERNEL void AttachBackendKernel(BackendT *pack) {
  HSHM_MEMORY_MANAGER;
  HSHM_THREAD_MODEL;
  HSHM_SYSTEM_INFO;
  HSHM_MEMORY_MANAGER->RegisterBackend(pack);
  HSHM_MEMORY_MANAGER->ScanBackends();
}
#endif

#ifdef HSHM_ENABLE_CUDA
/** Check if a backend is cuda-compatible */
void AllocateCudaBackend(int dev, MemoryBackend *other) {
  cudaSetDevice(dev);
  switch (other->header_->type_) {
    case MemoryBackendType::kCudaMalloc: {
      // CudaMalloc *pack, *cpy;
      // cudaMallocManaged(&pack, sizeof(CudaMalloc));
      // memcpy((char *)pack, (char *)other, sizeof(CudaMalloc));
      // pack->UnsetScanned();
      // pack->Disown();
      // AttachBackendKernel<<<1, 1>>>(pack);
      // cudaDeviceSynchronize();
      // cudaFree(pack);
      break;
    }
    case MemoryBackendType::kCudaShmMmap: {
      CudaShmMmap *pack, *cpy;
      cudaMallocManaged(&pack, sizeof(CudaShmMmap));
      memcpy((char *)pack, (char *)other, sizeof(CudaShmMmap));
      pack->UnsetScanned();
      pack->Disown();
      AttachBackendKernel<<<1, 1>>>(pack);
      cudaDeviceSynchronize();
      cudaFree(pack);
      break;
    }
    default: {
      break;
    }
  }
}
#endif

#ifdef HSHM_ENABLE_ROCM
/** Check if a backend is cuda-compatible */
void AllocateRocmBackend(int dev, MemoryBackend *other) {
  HIP_ERROR_CHECK(hipSetDevice(dev));
  switch (other->header_->type_) {
    case MemoryBackendType::kRocmMalloc: {
      // RocmMalloc *pack, *cpy;
      // HIP_ERROR_CHECK(hipMallocManaged(&pack, sizeof(RocmMalloc)));
      // memcpy((char *)pack, (char *)other, sizeof(RocmMalloc));
      // pack->UnsetScanned();
      // pack->Disown();
      // AttachBackendKernel<<<1, 1>>>(pack);
      // HIP_ERROR_CHECK(hipDeviceSynchronize());
      break;
    }
    case MemoryBackendType::kRocmShmMmap: {
      RocmShmMmap *pack, *cpy;
      HIP_ERROR_CHECK(hipMallocManaged(&pack, sizeof(RocmShmMmap)));
      memcpy((char *)pack, (char *)other, sizeof(RocmShmMmap));
      pack->UnsetScanned();
      pack->Disown();
      AttachBackendKernel<<<1, 1>>>(pack);
      HIP_ERROR_CHECK(hipDeviceSynchronize());
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
  int num_gpus = 0;
#if defined(HSHM_ENABLE_CUDA) && defined(HSHM_IS_HOST)
  cudaGetDeviceCount(&num_gpus);
#endif
#if defined(HSHM_ENABLE_ROCM) && defined(HSHM_IS_HOST)
  HIP_ERROR_CHECK(hipGetDeviceCount(&num_gpus));
#endif
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
#if defined(HSHM_ENABLE_CUDA) && defined(HSHM_IS_HOST)
    for (int dev = 0; dev < num_gpus; ++dev) {
      AllocateCudaBackend(dev, backend);
    }
#endif
#if defined(HSHM_ENABLE_ROCM) && defined(HSHM_IS_HOST)
    for (int dev = 0; dev < num_gpus; ++dev) {
      AllocateRocmBackend(dev, backend);
    }
#endif
  }
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

HSHM_DEFINE_GLOBAL_PTR_VAR_CC(hshm::ipc::MemoryManager, hshmMemoryManager);

}  // namespace hshm::ipc

// TODO(llogan): Fix. A hack for HIP compiler to function
// I would love to spend more time figuring out why ROCm
// Fails to compile without this, but whatever.
#include "hermes_shm/introspect/system_info.cc"
