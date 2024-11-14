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


#ifndef HERMES_SHM_CONTAINER_H_
#define HERMES_SHM_CONTAINER_H_

#include "hermes_shm/memory/memory_manager_.h"
#include "hermes_shm/constants/macros.h"
#include "hermes_shm/types/bitfield.h"
#include "shm_macros.h"

namespace hshm::ipc {

/**
 * ShmContainers all have a header, which is stored in
 * shared memory as a TypedPointer.
 * */
class ShmContainer {};

/**
 * Flags
 * */
typedef u32 ShmFlagField;
struct ShmFlag {
  CLS_CONST u32 kIsPrivate = BIT_OPT(u32, 0);
  CLS_CONST u32 kIsUndestructable = BIT_OPT(u32, 1);
  CLS_CONST u32 kIsThreadLocal = kIsPrivate | kIsUndestructable;
};

/**
 * Flags for ShmContainer classes
 * */
#define HSHM_CLASS_TEMPL_WITH_DEFAULTS \
  typename AllocT = HSHM_DEFAULT_ALLOC, ShmFlagField HSHM_FLAGS = 0
#define HSHM_CLASS_TEMPL \
  typename AllocT, ShmFlagField HSHM_FLAGS
#define HSHM_CLASS_TEMPL_ARGS \
  AllocT, HSHM_FLAGS
#define HSHM_CLASS_TEMPL_TLS_ARGS \
  AllocT, hipc::ShmFlag::kIsThreadLocal
#define HSHM_ALLOCATOR_INFO \
  typename std::conditional<HSHM_FLAGS & hipc::ShmFlag::kIsPrivate, \
                   hipc::CtxAllocator<AllocT>, \
                   hipc::AllocatorId>::type

/** Typed nullptr */
template<typename T>
HSHM_INLINE_CROSS_FUN static T* typed_nullptr() {
  return reinterpret_cast<T*>(NULL);
}

/**
 * The main container template macro
 * */
#define HIPC_CONTAINER_TEMPLATE(CLASS_NAME,CLASS_NEW_ARGS) \
public: \
/**==================================== \
 * Variables & Types \
 * ===================================*/ \
HSHM_ALLOCATOR_INFO alloc_info_; \
 \
/**==================================== \
 * Constructors \
 * ===================================*/ \
/** Get thread-local reference */ \
HSHM_CROSS_FUN \
TYPE_UNWRAP(CLASS_NAME)<TYPE_UNWRAP(CLASS_NEW_ARGS), HSHM_CLASS_TEMPL_TLS_ARGS> \
GetThreadLocal(const hipc::ScopedTlsAllocator<AllocT> &tls_alloc) { \
  return GetThreadLocal(tls_alloc.alloc_); \
} \
 \
/** Get thread-local reference */ \
HSHM_CROSS_FUN \
TYPE_UNWRAP(CLASS_NAME)<TYPE_UNWRAP(CLASS_NEW_ARGS), HSHM_CLASS_TEMPL_TLS_ARGS> \
GetThreadLocal(const hipc::CtxAllocator<AllocT> &ctx_alloc) { \
  return GetThreadLocal(ctx_alloc.ctx_.tid_); \
} \
 \
/** Get thread-local reference */ \
HSHM_CROSS_FUN \
TYPE_UNWRAP(CLASS_NAME)<TYPE_UNWRAP(CLASS_NEW_ARGS), HSHM_CLASS_TEMPL_TLS_ARGS> \
GetThreadLocal(const hshm::ThreadId &tid) { \
  return TYPE_UNWRAP(CLASS_NAME)<TYPE_UNWRAP(CLASS_NEW_ARGS), HSHM_CLASS_TEMPL_TLS_ARGS>( \
      *this, tid, GetAllocator()); \
} \
 \
 \
/** SHM constructor. Thread-local. */ \
template<hipc::ShmFlagField OTHER_FLAGS> \
HSHM_CROSS_FUN \
explicit TYPE_UNWRAP(CLASS_NAME)( \
    const TYPE_UNWRAP(CLASS_NAME)<TYPE_UNWRAP(CLASS_NEW_ARGS), AllocT, OTHER_FLAGS> &other, \
const hshm::ThreadId &tid, AllocT *alloc) { \
memcpy(this, &other, sizeof(*this)); \
init_shm_container(tid, alloc); \
} \
 \
/** Initialize container */ \
HSHM_CROSS_FUN \
void init_shm_container(AllocT *alloc) { \
  if constexpr (!(HSHM_FLAGS & hipc::ShmFlag::kIsPrivate)) { \
    alloc_info_ = alloc->GetId(); \
  } else { \
    alloc_info_.alloc_ = alloc; \
    alloc_info_.ctx_ = hipc::MemContext(); \
  } \
} \
 \
/** Initialize container (thread-local) */ \
HSHM_CROSS_FUN \
void init_shm_container(const hipc::MemContext &ctx, AllocT *alloc) { \
  if constexpr (!(HSHM_FLAGS & hipc::ShmFlag::kIsPrivate)) { \
    alloc_info_ = alloc->GetId(); \
  } else { \
    alloc_info_.alloc_ = alloc; \
    alloc_info_.ctx_ = ctx; \
  } \
} \
 \
/** Initialize container (thread-local) */ \
HSHM_CROSS_FUN \
void init_shm_container(const hipc::CtxAllocator<AllocT> &tls_alloc) { \
  init_shm_container(tls_alloc.ctx_, tls_alloc.alloc_); \
} \
 \
/**==================================== \
 * Destructor \
 * ===================================*/ \
/** Destructor. */ \
HSHM_INLINE_CROSS_FUN \
~TYPE_UNWRAP(CLASS_NAME)() { \
  if constexpr ((HSHM_FLAGS & hipc::ShmFlag::kIsUndestructable)) { \
    shm_destroy(); \
  } \
} \
 \
/** Destruction operation */ \
HSHM_INLINE_CROSS_FUN \
void shm_destroy() { \
  if (IsNull()) { return; } \
  shm_destroy_main(); \
  SetNull(); \
} \
 \
/**==================================== \
 * Header Operations \
 * ===================================*/ \
 \
/** Get a typed pointer to the object */ \
template<typename POINTER_T> \
HSHM_INLINE_CROSS_FUN \
    POINTER_T GetShmPointer() const { \
  return GetAllocator()->template \
      Convert<TYPE_UNWRAP(CLASS_NAME)<TYPE_UNWRAP(CLASS_NEW_ARGS), HSHM_CLASS_TEMPL_ARGS>, \
      POINTER_T>(this); \
} \
 \
/**==================================== \
 * Query Operations \
 * ===================================*/ \
 \
/** Get the allocator for this container */ \
HSHM_INLINE_CROSS_FUN \
    AllocT* GetAllocator() const { \
  if constexpr (!(HSHM_FLAGS & hipc::ShmFlag::kIsPrivate)) { \
    return (AllocT*)HERMES_MEMORY_MANAGER->GetAllocator(alloc_info_); \
  } else { \
    return alloc_info_.alloc_; \
  } \
} \
 \
/** Get the shared-memory allocator id */ \
HSHM_INLINE_CROSS_FUN \
const hipc::AllocatorId& GetAllocatorId() const { \
  if constexpr (!(HSHM_FLAGS & hipc::ShmFlag::kIsPrivate)) { \
    return alloc_info_; \
  } else { \
    return GetAllocator()->GetId(); \
  } \
} \
 \
/** Get the shared-memory allocator id */ \
HSHM_INLINE_CROSS_FUN \
    hshm::ThreadId GetThreadId() const { \
  if constexpr (!(HSHM_FLAGS & hipc::ShmFlag::kIsPrivate)) { \
    return hshm::ThreadId::GetNull(); \
  } else { \
    return alloc_info_.ctx_; \
  } \
} \
 \
/** Get the shared-memory allocator id */ \
HSHM_INLINE_CROSS_FUN \
    hipc::CtxAllocator<AllocT> GetCtxAllocator() const { \
  return hipc::CtxAllocator<AllocT>{GetThreadId(), GetAllocator()}; \
}


}  // namespace hshm::ipc

#endif  // HERMES_SHM_CONTAINER_H_
