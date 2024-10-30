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
#include "shm_macros.h"

namespace hshm::ipc {

/** The shared-memory header used for data structures */
template<typename T>
struct ShmHeader;

/**
 * ShmContainers all have a header, which is stored in
 * shared memory as a TypedPointer.
 * */
class ShmContainer {};

/**
 * Flags for ShmContainer classes
 * */
#define HSHM_CLASS_TEMPL_WITH_DEFAULTS \
  typename AllocT = HSHM_DEFAULT_ALLOC, bool IsPrivate = false
#define HSHM_CLASS_TEMPL \
  typename AllocT, bool IsPrivate
#define HSHM_CLASS_TEMPL_ARGS \
  AllocT, IsPrivate
#define HSHM_ALLOCATOR_INFO \
  typename std::conditional<IsPrivate, \
                   AllocT, \
                   hipc::AllocatorId>::type

/** The ShmHeader used for wrapper containers */
struct ShmWrapperHeader {};

/** Typed nullptr */
template<typename T>
HSHM_INLINE_CROSS_FUN static T* typed_nullptr() {
  return reinterpret_cast<T*>(NULL);
}

/**
 * The main container template macro
 * */
#define HIPC_CONTAINER_TEMPLATE(CLASS_NAME,TYPED_CLASS) \
public:\
/**====================================\
 * Variables & Types\
 * ===================================*/\
HSHM_ALLOCATOR_INFO alloc_info_;\
\
/**====================================\
 * Constructors\
 * ===================================*/\
/** Initialize container */\
HSHM_CROSS_FUN void init_shm_container(AllocT *alloc) {\
  if constexpr (!IsPrivate) {\
    alloc_info_ = alloc->GetId();\
  } else {\
    alloc_info_ = alloc;\
  }\
}\
\
/**====================================\
 * Destructor\
 * ===================================*/\
/** Destructor. */\
HSHM_INLINE_CROSS_FUN ~TYPE_UNWRAP(CLASS_NAME)() {\
  shm_destroy();\
}\
\
/** Destruction operation */\
HSHM_INLINE_CROSS_FUN void shm_destroy() {\
  if (IsNull()) { return; }\
  shm_destroy_main();\
  SetNull();\
}\
\
/**====================================\
 * Header Operations\
 * ===================================*/\
\
/** Get a typed pointer to the object */\
template<typename POINTER_T>\
HSHM_INLINE_CROSS_FUN POINTER_T GetShmPointer() const {\
  return GetAllocator()->template Convert<TYPE_UNWRAP(TYPED_CLASS), POINTER_T>(this);\
}\
\
/**====================================\
 * Query Operations\
 * ===================================*/\
\
/** Get the allocator for this container */\
HSHM_INLINE_CROSS_FUN AllocT* GetAllocator() const {\
  if constexpr (!IsPrivate) {\
    return (AllocT*)HERMES_MEMORY_MANAGER->GetAllocator(alloc_info_);\
  } else {\
    return alloc_info_;\
  }\
}\
\
/** Get the shared-memory allocator id */\
HSHM_INLINE_CROSS_FUN const hipc::AllocatorId& GetAllocatorId() const {\
  if constexpr (!IsPrivate) {\
    return alloc_info_;\
  } else {\
    return GetAllocator()->GetId();\
  }\
}

}  // namespace hshm::ipc

#endif  // HERMES_SHM_CONTAINER_H_
