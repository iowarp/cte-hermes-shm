//
// Created by llogan on 10/11/24.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_MEMORY_MEMORY_MANAGER__H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_MEMORY_MEMORY_MANAGER__H_

#include "hermes_shm/memory/allocator/allocator.h"
#include "hermes_shm/memory/backend/posix_mmap.h"
#include "hermes_shm/util/singleton/_global_singleton.h"


#define HERMES_MEMORY_MANAGER \
  hshm::GlobalSingleton<hshm::ipc::MemoryManager>::GetInstance()
#define HERMES_MEMORY_MANAGER_T hshm::ipc::MemoryManager*

namespace hipc = hshm::ipc;

namespace hshm::ipc {

#define MAX_ALLOCATORS 64

class MemoryManager {
 public:
  allocator_id_t root_allocator_id_;
  MemoryBackend *root_backend_;
  Allocator *root_allocator_;
  void *backends_;
  Allocator *allocators_[MAX_ALLOCATORS];
  Allocator *default_allocator_;

 public:
  /** Create the root allocator */
  HSHM_CROSS_FUN
  MemoryManager();

  /** Default backend size */
  HSHM_CROSS_FUN
  static size_t GetDefaultBackendSize();

  /**
   * Create a memory backend. Memory backends are divided into slots.
   * Each slot corresponds directly with a single allocator.
   * There can be multiple slots per-backend, enabling multiple allocation
   * policies over a single memory region.
   * */
  template<typename BackendT, typename ...Args>
  HSHM_CROSS_FUN
  MemoryBackend* CreateBackend(size_t size,
                               const hshm::chararr &url,
                               Args&& ...args);

  /**
   * Register a unique memory backend. Throws an exception if the backend
   * already exists. This is because unregistering a backend can cause
   * ramifications across allocators.
   *
   * @param url the backend's unique identifier
   * @param backend the backend to register
   * */
  HSHM_CROSS_FUN
  MemoryBackend* RegisterBackend(
      const hshm::chararr &url,
      MemoryBackend* backend);

  /**
   * Attaches to an existing memory backend located at \a url url.
   * */
  HSHM_CROSS_FUN
  MemoryBackend* AttachBackend(MemoryBackendType type,
                               const hshm::chararr &url);

  /**
   * Returns a pointer to a backend that has already been attached.
   * */
  HSHM_CROSS_FUN
  MemoryBackend* GetBackend(const hshm::chararr &url);

  /**
   * Unregister backend
   * */
  HSHM_CROSS_FUN
  void UnregisterBackend(const hshm::chararr &url);

  /**
   * Destroy backend
   * */
  HSHM_CROSS_FUN
  void DestroyBackend(const hshm::chararr &url);

  /**
   * Scans all attached backends for new memory allocators.
   * */
  HSHM_CROSS_FUN
  void ScanBackends();

  /**
   * Create and register a memory allocator for a particular backend.
   * */
  template<typename AllocT, typename ...Args>
  HSHM_CROSS_FUN
  Allocator* CreateAllocator(const hshm::chararr &url,
                             allocator_id_t alloc_id,
                             size_t custom_header_size,
                             Args&& ...args);

  /**
   * Registers an allocator. Used internally by ScanBackends, but may
   * also be used externally.
   * */
  HSHM_CROSS_FUN
  Allocator* RegisterAllocator(Allocator *alloc);

  /**
   * Destroys an allocator
   * */
  HSHM_CROSS_FUN
  void UnregisterAllocator(allocator_id_t alloc_id);

  /**
   * Locates an allocator of a particular id
   * */
  HSHM_CROSS_FUN Allocator* GetAllocator(allocator_id_t alloc_id);

  /**
   * Gets the allocator used for initializing other allocators.
   * */
  HSHM_CROSS_FUN Allocator* GetRootAllocator();

  /**
   * Gets the allocator used by default when no allocator is
   * used to construct an object.
   * */
  HSHM_CROSS_FUN Allocator* GetDefaultAllocator();

  /**
   * Sets the allocator used by default when no allocator is
   * used to construct an object.
   * */
  HSHM_CROSS_FUN void SetDefaultAllocator(Allocator *alloc);

  /**
   * Convert a process-independent pointer into a process-specific pointer.
   * */
  template<typename T, typename POINTER_T = Pointer>
  HSHM_INLINE_CROSS_FUN T* Convert(const POINTER_T &p) {
    if (p.IsNull()) {
      return nullptr;
    }
    return GetAllocator(p.allocator_id_)->template
        Convert<T, POINTER_T>(p);
  }

  /**
   * Convert a process-specific pointer into a process-independent pointer
   *
   * @param allocator_id the allocator the pointer belongs to
   * @param ptr the pointer to convert
   * */
  template<typename T, typename POINTER_T = Pointer>
  HSHM_INLINE_CROSS_FUN POINTER_T Convert(allocator_id_t allocator_id, T *ptr) {
    return GetAllocator(allocator_id)->template
        Convert<T, POINTER_T>(ptr);
  }

  /**
   * Convert a process-specific pointer into a process-independent pointer when
   * the allocator is unkown.
   *
   * @param ptr the pointer to convert
   * */
  template<typename T, typename POINTER_T = Pointer>
  HSHM_INLINE_CROSS_FUN POINTER_T Convert(T *ptr) {
    for (auto &alloc : HERMES_MEMORY_MANAGER->allocators_) {
      if (alloc && alloc->ContainsPtr(ptr)) {
        return alloc->template
            Convert<T, POINTER_T>(ptr);
      }
    }
    return Pointer::GetNull();
  }
};

}  // namespace hshm::ipc


#endif //HERMES_SHM_INCLUDE_HERMES_SHM_MEMORY_MEMORY_MANAGER__H_
