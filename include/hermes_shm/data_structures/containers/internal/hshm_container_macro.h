#ifndef HERMES_DATA_STRUCTURES_CONTAINER_MACRO_H_
#define HERMES_DATA_STRUCTURES_CONTAINER_MACRO_H_
#include "hermes_shm/constants/macros.h"
#define HSHM_CONTAINER_BASE_TEMPLATE \
public:\
/**====================================\
 * Variables & Types\
 * ===================================*/\
hipc::Allocator *alloc_;\
\
/**====================================\
 * Query Operations\
 * ===================================*/\
\
/** Get the allocator for this container */\
HSHM_INLINE_CROSS_FUN hipc::Allocator* GetAllocator() const {\
  return alloc_;\
}\
\
/** Get the shared-memory allocator id */\
HSHM_INLINE_CROSS_FUN hipc::allocator_id_t& GetAllocatorId() const {\
  return GetAllocator()->GetId();\
}

#endif  // HERMES_DATA_STRUCTURES_CONTAINER_MACRO_H_
