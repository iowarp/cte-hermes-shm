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

#ifndef HERMES_INCLUDE_HERMES_DATA_STRUCTURES_INTERNAL_SHM_CONTAINER_EXAMPLE_H_
#define HERMES_INCLUDE_HERMES_DATA_STRUCTURES_INTERNAL_SHM_CONTAINER_EXAMPLE_H_

#include "hermes_shm/data_structures/ipc/internal/shm_container.h"
#include "hermes_shm/memory/memory_manager.h"

namespace honey {

#define CLASS_NAME ShmContainerExample
#define TYPED_CLASS ShmContainerExample

template<typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
class ShmContainerExample : public hipc::ShmContainer {
 public:
  /**====================================
   * Variables & Types
   * ===================================*/
  HSHM_ALLOCATOR_INFO alloc_info_;

  /**====================================
   * Constructors
   * ===================================*/
  /** Initialize container */
  HSHM_CROSS_FUN
  void init_shm_container(AllocT *alloc) {
    if constexpr (!(HSHM_FLAGS & hipc::ShmFlag::kIsPrivate)) {
      alloc_info_ = alloc->GetId();
    } else {
      alloc_info_.alloc_ = alloc;
      alloc_info_.tls_ = hshm::ThreadId::GetNull();
    }
  }

  /** Initialize container (thread-local) */
  HSHM_CROSS_FUN
  void init_shm_container(const hshm::ThreadId &tid, AllocT *alloc) {
    if constexpr (!(HSHM_FLAGS & hipc::ShmFlag::kIsPrivate)) {
      alloc_info_ = alloc->GetId();
    } else {
      alloc_info_.alloc_ = alloc;
      alloc_info_.tls_ = tid;
    }
  }

  /** Initialize container (thread-local) */
  HSHM_CROSS_FUN
  void init_shm_container(const hipc::TlsAllocator<AllocT> &tls_alloc) {
    init_shm_container(tls_alloc.tls_, tls_alloc.alloc_);
  }

  /**====================================
   * Destructor
   * ===================================*/
  /** Destructor. */
  HSHM_INLINE_CROSS_FUN
  ~TYPE_UNWRAP(CLASS_NAME)() {
    if constexpr ((HSHM_FLAGS & hipc::ShmFlag::kIsUndestructable)) {
      shm_destroy();
    }
  }

  /** Destruction operation */
  HSHM_INLINE_CROSS_FUN
  void shm_destroy() {
    if (IsNull()) { return; }
    shm_destroy_main();
    SetNull();
  }

  /**====================================
   * Header Operations
   * ===================================*/

  /** Get a typed pointer to the object */
  template<typename POINTER_T>
  HSHM_INLINE_CROSS_FUN
  POINTER_T GetShmPointer() const {
    return GetAllocator()->template Convert<TYPE_UNWRAP(TYPED_CLASS), POINTER_T>(this);
  }

  /**====================================
   * Query Operations
   * ===================================*/

  /** Get the allocator for this container */
  HSHM_INLINE_CROSS_FUN
  AllocT* GetAllocator() const {
    if constexpr (!(HSHM_FLAGS & hipc::ShmFlag::kIsPrivate)) {
      return (AllocT*)HERMES_MEMORY_MANAGER->GetAllocator(alloc_info_);
    } else {
      return alloc_info_.alloc_;
    }
  }

  /** Get the shared-memory allocator id */
  HSHM_INLINE_CROSS_FUN
  const hipc::AllocatorId& GetAllocatorId() const {
    if constexpr (!(HSHM_FLAGS & hipc::ShmFlag::kIsPrivate)) {
      return alloc_info_;
    } else {
      return GetAllocator()->GetId();
    }
  }

  /** Get the shared-memory allocator id */
  HSHM_INLINE_CROSS_FUN
  hshm::ThreadId GetThreadId() const {
    if constexpr (!(HSHM_FLAGS & hipc::ShmFlag::kIsPrivate)) {
      return hshm::ThreadId::GetNull();
    } else {
      return alloc_info_.tls_;
    }
  }

  /** Get the shared-memory allocator id */
  HSHM_INLINE_CROSS_FUN
  hipc::TlsAllocator<AllocT> GetTlsAllocator() const {
    return hipc::TlsAllocator<AllocT>{GetThreadId(), GetAllocator()};
  }
};

}  // namespace hshm::ipc

#undef CLASS_NAME
#undef TYPED_CLASS

#endif  // HERMES_INCLUDE_HERMES_DATA_STRUCTURES_INTERNAL_SHM_CONTAINER_EXAMPLE_H_
