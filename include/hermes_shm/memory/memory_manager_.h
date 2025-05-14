//
// Created by llogan on 10/11/24.
//

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_MEMORY_MEMORY_MANAGER__H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_MEMORY_MEMORY_MANAGER__H_

#include <cstddef>
#include <limits>

#include "allocator/allocator_factory_.h"
#include "hermes_shm/memory/allocator/allocator.h"
#include "hermes_shm/memory/backend/posix_mmap.h"
#include "hermes_shm/types/numbers.h"
#include "hermes_shm/util/gpu_api.h"
#include "hermes_shm/util/singleton.h"

namespace hshm::ipc {

/** Max # of GPUs mem mngr can hold */
#define HSHM_MAX_GPUS 16

/** Max # of allocator the mem mngr can hold */
#define HSHM_MAX_ALLOCATORS 64

/** Max number of memory backends that can be mounted */
#define HSHM_MAX_BACKENDS 16

/** Memory manager class */
class MemoryManager {
 public:
  char root_alloc_data_[hshm::Unit<size_t>::Kilobytes(64)];
  char root_backend_space_[256];
  char root_alloc_space_[256];
  AllocatorId root_alloc_id_;
  MemoryBackend *root_backend_;
  Allocator *root_alloc_;
  MemoryManager **gpu_ptrs_[HSHM_MAX_GPUS];
  MemoryBackend *backends_[HSHM_MAX_BACKENDS];
  Allocator *allocators_[HSHM_MAX_ALLOCATORS];
  Allocator *default_allocator_;

 public:
  /** Create the root allocator */
  HSHM_CROSS_FUN
  MemoryManager() { Init(); }

  /**
   * Initialize memory manager
   * Automatically called in default constructor if on CPU.
   * Must be called explicitly if on GPU.
   * */
  template <int nothing = 0>
  HSHM_CROSS_FUN void Init();

  /** Default backend size */
  template <int nothing = 0>
  HSHM_CROSS_FUN static size_t GetDefaultBackendSize();

  /**
   * Create a memory backend. Memory backends are divided into slots.
   * Each slot corresponds directly with a single allocator.
   * There can be multiple slots per-backend, enabling multiple allocation
   * policies over a single memory region.
   * */
  template <typename BackendT, typename... Args>
  MemoryBackend *CreateBackend(const MemoryBackendId &backend_id, size_t size,
                               Args &&...args);

  /**
   * Create a memory backend. Always includes the url parameter.
   * */
  template <typename BackendT, typename... Args>
  MemoryBackend *CreateBackendWithUrl(const MemoryBackendId &backend_id,
                                      size_t size, const hshm::chararr &url,
                                      Args &&...args) {
    if constexpr (std::is_base_of_v<UrlMemoryBackend, BackendT>) {
      return CreateBackend<BackendT>(backend_id, size, url,
                                     std::forward<Args>(args)...);
    } else {
      return CreateBackend<BackendT>(backend_id, size,
                                     std::forward<Args>(args)...);
    }
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
  MemoryBackend *RegisterBackend(MemoryBackend *backend) {
    if (GetBackend(backend->GetId())) {
      HSHM_THROW_ERROR(MEMORY_BACKEND_REPEATED);
    }
    backends_[backend->GetId().id_] = backend;
    return backend;
  }

  /** Copy backends to GPU */
  template <int nothing = 0>
  HSHM_INLINE void CopyAllBackendsToGpu();

  /** Copy and existing backend to the GPU */
  template <int nothing = 0>
  void CopyBackendGpu(const MemoryBackendId &backend_id);

  /** Create an array backend on the GPU */
  template <int nothing = 0>
  void CreateBackendGpu(int gpu_id, const MemoryBackendId &backend_id,
                        char *accel_data, size_t accel_data_size);

  /** Mark a backend as allocated on GPU */
  template <int nothing = 0>
  void SetBackendHasAllocGpu(const MemoryBackendId &backend_id);

  /**
   * Attaches to an existing memory backend located at \a url url.
   * */
  template <int nothing = 0>
  HSHM_CROSS_FUN MemoryBackend *AttachBackend(MemoryBackendType type,
                                              const hshm::chararr &url);

  /**
   * Returns a pointer to a backend that has already been attached.
   * */
  HSHM_CROSS_FUN
  MemoryBackend *GetBackend(const MemoryBackendId &backend_id) {
    return backends_[backend_id.id_];
  }

  /**
   * Unregister backend
   * */
  HSHM_CROSS_FUN
  MemoryBackend *UnregisterBackend(const MemoryBackendId &backend_id) {
    auto backend = backends_[backend_id.id_];
    backends_[backend_id.id_] = nullptr;
    return backend;
  }

  /** Free the backend */
  template <int nothing = 0>
  HSHM_CROSS_FUN void DestroyBackend(const MemoryBackendId &backend_id);

  /**
   * Scans all attached backends for new memory allocators.
   * */
  template <int nothing = 0>
  HSHM_CROSS_FUN void ScanBackends();

  /**
   * Scans all backends on all GPUs for new memory allocators
   */
  template <int nothing = 0>
  HSHM_INLINE void ScanBackendsAllGpu();

  /**
   * Scans all attached backends on GPU for new memory allocators.
   * */
  template <int nothing = 0>
  HSHM_INLINE void ScanBackendsGpu(int gpu_id);

  /**
   * Create and register a memory allocator for a particular backend.
   * */
  template <typename AllocT, typename... Args>
  HSHM_INLINE_CROSS_FUN AllocT *CreateAllocator(
      const MemoryBackendId &backend_id, const AllocatorId &alloc_id,
      size_t custom_header_size, Args &&...args);

  /** Create an allocator for the GPU */
  template <typename AllocT, typename... Args>
  void CreateAllocatorGpu(int gpu_id, const MemoryBackendId &backend_id,
                          const AllocatorId &alloc_id,
                          size_t custom_header_size, Args &&...args);

  /** Create an allocator for the GPU */
  template <typename AllocT>
  AllocT *ShareAllocatorGpu(int gpu_id, const AllocatorId &alloc_id);

  /**
   * Registers an allocator. Used internally by ScanBackends, but may
   * also be used externally.
   * */
  template <int nothing = 0>
  HSHM_CROSS_FUN Allocator *RegisterAllocator(Allocator *alloc);

  /**
   * Registers an allocator. Used internally by ScanBackends, but may
   * also be used externally.
   * */
  HSHM_CROSS_FUN
  Allocator *RegisterAllocatorNoScan(Allocator *alloc) {
    if (alloc == nullptr) {
      return nullptr;
    }
    if (default_allocator_ == nullptr || default_allocator_ == root_alloc_) {
      if (alloc->GetId().bits_.minor_ == 0) {
        default_allocator_ = alloc;
      }
    }
    uint32_t idx = alloc->GetId().ToIndex();
    if (idx > HSHM_MAX_ALLOCATORS) {
      HILOG(kError, "Allocator index out of range: {}", idx);
      HSHM_THROW_ERROR(TOO_MANY_ALLOCATORS);
    }
    allocators_[idx] = alloc;
    return alloc;
  }

  /**
   * Registers an internal allocator.
   * */
  HSHM_CROSS_FUN
  Allocator *RegisterSubAllocator(Allocator *alloc) {
    return RegisterAllocatorNoScan(alloc);
  }

  /**
   * Destroys an allocator
   * */
  HSHM_CROSS_FUN
  Allocator *UnregisterAllocator(const AllocatorId &alloc_id) {
    if (alloc_id == default_allocator_->GetId()) {
      default_allocator_ = root_alloc_;
    }
    auto alloc = allocators_[alloc_id.ToIndex()];
    allocators_[alloc_id.ToIndex()] = nullptr;
    return alloc;
  }

  template <typename AllocT>
  HSHM_CROSS_FUN void DestroyAllocator(const AllocatorId &alloc_id);

  /**
   * Locates an allocator of a particular id
   * */
  template <typename AllocT>
  HSHM_CROSS_FUN AllocT *GetAllocator(const AllocatorId &alloc_id) {
    return (AllocT *)allocators_[alloc_id.ToIndex()];
  }

  /**
   * Gets the allocator used for initializing other allocators.
   * */
  template <typename AllocT>
  HSHM_CROSS_FUN AllocT *GetRootAllocator() {
    return (AllocT *)root_alloc_;
  }

  /**
   * Gets the allocator used by default when no allocator is
   * used to construct an object.
   * */
  template <typename AllocT>
  HSHM_CROSS_FUN AllocT *GetDefaultAllocator() {
    return (AllocT *)(default_allocator_);
  }

  /**
   * Sets the allocator used by default when no allocator is
   * used to construct an object.
   * */
  template <typename AllocT>
  HSHM_CROSS_FUN void SetDefaultAllocator(AllocT *alloc) {
    default_allocator_ = alloc;
  }

  /**
   * Convert a process-independent pointer into a process-specific pointer.
   * */
  template <typename T, typename POINTER_T = Pointer>
  HSHM_INLINE_CROSS_FUN T *Convert(const POINTER_T &p) {
    if (p.IsNull()) {
      return nullptr;
    }
    return GetAllocator<NullAllocator>(p.alloc_id_)
        ->template Convert<T, POINTER_T>(p);
  }

  /**
   * Convert a process-specific pointer into a process-independent pointer
   *
   * @param allocator_id the allocator the pointer belongs to
   * @param ptr the pointer to convert
   * */
  template <typename T, typename POINTER_T = Pointer>
  HSHM_INLINE_CROSS_FUN POINTER_T Convert(AllocatorId allocator_id,
                                          const T *ptr) {
    return GetAllocator<NullAllocator>(allocator_id)
        ->template Convert<T, POINTER_T>(ptr);
  }

  /**
   * Convert a process-specific pointer into a process-independent pointer when
   * the allocator is unkown.
   *
   * @param ptr the pointer to convert
   * */
  template <typename T, typename POINTER_T = Pointer>
  HSHM_INLINE_CROSS_FUN POINTER_T Convert(T *ptr) {
    Allocator *best_alloc = FindNearestAllocator(ptr);
    if (best_alloc) {
      return best_alloc->template Convert<T, POINTER_T>(ptr);
    } else {
      return Pointer::GetNull();
    }
  }

  /**
   * Find the allocator that most closely contains the pointer
   */
  template <typename T>
  HSHM_INLINE_CROSS_FUN Allocator *FindNearestAllocator(T *ptr) {
    size_t range = std::numeric_limits<size_t>::max();
    Allocator *best_alloc = nullptr;
    for (auto &alloc : allocators_) {
      if (alloc && alloc->ContainsPtr(ptr) && alloc->buffer_size_ <= range) {
        range = alloc->buffer_size_;
        best_alloc = alloc;
      }
      if (range < std::numeric_limits<size_t>::max()) {
        break;
      }
    }
    return best_alloc;
  }
};

/** Singleton declaration */
HSHM_DEFINE_GLOBAL_CROSS_PTR_VAR_H(hshm::ipc::MemoryManager, hshmMemoryManager);
#define HSHM_MEMORY_MANAGER                               \
  HSHM_GET_GLOBAL_CROSS_PTR_VAR(hshm::ipc::MemoryManager, \
                                hshm::ipc::hshmMemoryManager)
#define HSHM_MEMORY_MANAGER_T hshm::ipc::MemoryManager *

}  // namespace hshm::ipc

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_MEMORY_MEMORY_MANAGER__H_
