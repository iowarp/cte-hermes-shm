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

class _TestAllocatorHeader : public AllocatorHeader {
 public:
  hipc::atomic<hshm::min_u64> total_alloc_;

  HSHM_CROSS_FUN
  _TestAllocatorHeader() = default;

  HSHM_CROSS_FUN
  void Configure(AllocatorId alloc_id, size_t custom_header_size) {
    AllocatorHeader::Configure(alloc_id, AllocatorType::kThreadLocalAllocator,
                               custom_header_size);
    total_alloc_ = 0;
  }
};

class _TestAllocator : public Allocator {
 public:
  HSHM_ALLOCATOR(_TestAllocator);
  _TestAllocatorHeader *header_;

  /** Init */
  void shm_init(AllocatorId alloc_id, size_t custom_header_size, char *buffer,
                size_t buffer_size) {
    type_ = AllocatorType::kThreadLocalAllocator;
    id_ = alloc_id;
    buffer_ = buffer;
    buffer_size_ = buffer_size;
    header_ = reinterpret_cast<_TestAllocatorHeader *>(buffer_);
    custom_header_ = reinterpret_cast<char *>(header_ + 1);
    size_t region_off = (custom_header_ - buffer_) + custom_header_size;
    size_t region_size = buffer_size_ - region_off;
    header_->Configure(alloc_id, custom_header_size);
  }

  /**
   * Deserialize allocator from a buffer.
   * */
  HSHM_CROSS_FUN
  void shm_deserialize(char *buffer, size_t buffer_size) {
    buffer_ = buffer;
    buffer_size_ = buffer_size;
    header_ = reinterpret_cast<_TestAllocatorHeader *>(buffer_);
    type_ = header_->allocator_type_;
    id_ = header_->alloc_id_;
    custom_header_ = reinterpret_cast<char *>(header_ + 1);
    size_t region_off =
        (custom_header_ - buffer_) + header_->custom_header_size_;
    size_t region_size = buffer_size_ - region_off;
  }

  /**====================================
   * Core Allocator API
   * ===================================*/
 public:
  /**
   * Allocate a region of memory of \a size size
   * */
  HSHM_CROSS_FUN
  OffsetPointer AllocateOffset(const MemContext &ctx, size_t size) {
    header_->total_alloc_.fetch_add(1);
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
  void FreeOffsetNoNullCheck(const MemContext &ctx, OffsetPointer p) {
    header_->total_alloc_.fetch_sub(1);
  }

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