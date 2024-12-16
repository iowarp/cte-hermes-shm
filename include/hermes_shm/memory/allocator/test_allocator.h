#ifndef HERMES_SHM_MEMORY_ALLOCATOR_TEST_ALLOCATOR_H
#define HERMES_SHM_MEMORY_ALLOCATOR_TEST_ALLOCATOR_H

#include <cmath>

#include "allocator.h"
#include "hermes_shm/data_structures/ipc/list.h"
#include "hermes_shm/data_structures/ipc/pair.h"
#include "hermes_shm/data_structures/ipc/ring_ptr_queue.h"
#include "hermes_shm/data_structures/ipc/ring_queue.h"
#include "hermes_shm/data_structures/ipc/vector.h"
#include "hermes_shm/memory/allocator/stack_allocator.h"
#include "hermes_shm/thread/lock.h"
#include "hermes_shm/util/logging.h"
#include "hermes_shm/util/timer.h"
#include "mp_page.h"
#include "page_allocator.h"

namespace hshm::ipc {

class _TestAllocator;
typedef BaseAllocator<_TestAllocator> TestAllocator;

class _TestAllocator : public Allocator {
 public:
  /**====================================
   * Constructors
   * ===================================*/
  /**
   * Create the shared-memory allocator with \a id unique allocator id over
   * the particular slot of a memory backend.
   *
   * The shm_init function is required, but cannot be marked virtual as
   * each allocator has its own arguments to this method. Though each
   * allocator must have "id" as its first argument.
   * */
  void shm_init(AllocatorId alloc_id, size_t custom_header_size, char *buffer,
                size_t buffer_size) {}

  /**
   * Deserialize allocator from a buffer.
   * */
  HSHM_CROSS_FUN
  void shm_deserialize(char *buffer, size_t buffer_size) {}

  /**====================================
   * Core Allocator API
   * ===================================*/
 public:
  /**
   * Allocate a region of memory of \a size size
   * */
  HSHM_CROSS_FUN
  OffsetPointer AllocateOffset(const MemContext &ctx, size_t size) {
    return OffsetPointer::GetNull();
  }

  /**
   * Allocate a region of memory of \a size size
   * and \a alignment alignment. Assumes that
   * alignment is not 0.
   * */
  HSHM_CROSS_FUN
  OffsetPointer AlignedAllocateOffset(const MemContext &ctx, size_t size,
                                      size_t alignment) {
    return OffsetPointer::GetNull();
  }

  /**
   * Reallocate \a pointer to \a new_size new size.
   * Assumes that p is not kNulFullPtr.
   *
   * @return true if p was modified.
   * */
  HSHM_CROSS_FUN
  OffsetPointer ReallocateOffsetNoNullCheck(const MemContext &ctx,
                                            OffsetPointer p, size_t new_size) {
    return p;
  }

  /**
   * Free the memory pointed to by \a ptr Pointer
   * */
  HSHM_CROSS_FUN
  void FreeOffsetNoNullCheck(const MemContext &ctx, OffsetPointer p) {}

  /**
   * Create a globally-unique thread ID
   * */
  HSHM_CROSS_FUN
  void CreateTls(MemContext &ctx) {}

  /**
   * Free the memory pointed to by \a ptr Pointer
   * */
  HSHM_CROSS_FUN
  void FreeTls(const MemContext &ctx) {}

  /**
   * Get the amount of memory that was allocated, but not yet freed.
   * Useful for memory leak checks.
   * */
  HSHM_CROSS_FUN
  size_t GetCurrentlyAllocatedSize() { return 0; }
};

}  // namespace hshm::ipc

#endif  // HERMES_SHM_MEMORY_ALLOCATOR_TEST_ALLOCATOR_H