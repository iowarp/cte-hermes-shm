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


#ifndef HERMES_MEMORY_ALLOCATOR_SCALABLE_PAGE_ALLOCATOR_H
#define HERMES_MEMORY_ALLOCATOR_SCALABLE_PAGE_ALLOCATOR_H

#include "allocator.h"
#include "hermes_shm/thread/lock.h"
#include "hermes_shm/data_structures/pair.h"
#include "hermes_shm/data_structures/thread_unsafe/vector.h"
#include "hermes_shm/data_structures/thread_unsafe/list.h"
#include "hermes_shm/data_structures/pair.h"
#include <hermes_shm/memory/allocator/stack_allocator.h>
#include "mp_page.h"

namespace hermes::ipc {

struct FreeListStats {
  size_t page_size_;  /**< Page size stored in this free list */
  size_t count_;      /**< Number of allocations before coalesce */
};

struct ScalablePageAllocatorHeader : public AllocatorHeader {
  ShmArchiveOrT<vector<pair<FreeListStats, iqueue<MpPage>>>> free_lists_;
  std::atomic<size_t> total_alloc_;
  size_t coalesce_trigger_;
  size_t coalesce_window_;
  RwLock coalesce_lock_;

  ScalablePageAllocatorHeader() = default;

  void Configure(allocator_id_t alloc_id,
                 size_t custom_header_size,
                 Allocator *alloc,
                 size_t buffer_size,
                 RealNumber coalesce_trigger,
                 size_t coalesce_window) {
    AllocatorHeader::Configure(alloc_id,
                               AllocatorType::kScalablePageAllocator,
                               custom_header_size);
    free_lists_.shm_init(alloc);
    total_alloc_ = 0;
    coalesce_trigger_ = (coalesce_trigger * buffer_size).as_int();
    coalesce_window_ = coalesce_window;
    coalesce_lock_.Init();
  }
};

class ScalablePageAllocator : public Allocator {
 private:
  ScalablePageAllocatorHeader *header_;
  hipc::ShmRef<vector<pair<FreeListStats, iqueue<MpPage>>>> free_lists_;
  StackAllocator alloc_;
  /**
   * Cache every size between 16 (2^4) BYTES and 16KB (2^14): (11 entries)
   * */
  static const size_t num_caches_ = 11;
  /**
   * Cache every size between 16 (2^4) BYTES and 16KB (2^14): (11 entries)
   * Store every size larger than 16KB exactly in a free list. (1 entry)
   * Total of 12 entries.
   * */
  static const size_t num_free_lists_ = 12;

 public:
  /**
   * Allocator constructor
   * */
  ScalablePageAllocator()
  : header_(nullptr) {}

  /**
   * Get the ID of this allocator from shared memory
   * */
  allocator_id_t GetId() override {
    return header_->allocator_id_;
  }

  /**
   * Initialize the allocator in shared memory
   * */
  void shm_init(allocator_id_t id,
                size_t custom_header_size,
                char *buffer,
                size_t buffer_size,
                RealNumber coalesce_trigger = RealNumber(1, 5),
                size_t coalesce_window = MEGABYTES(1));

  /**
   * Attach an existing allocator from shared memory
   * */
  void shm_deserialize(char *buffer,
                       size_t buffer_size) override;

  /**
   * Allocate a memory of \a size size. The page allocator cannot allocate
   * memory larger than the page size.
   * */
  OffsetPointer AllocateOffset(size_t size) override;

  /**
   * Allocate a memory of \a size size, which is aligned to \a
   * alignment.
   * */
  OffsetPointer AlignedAllocateOffset(size_t size, size_t alignment) override;

  /**
   * Reallocate \a p pointer to \a new_size new size.
   *
   * @return whether or not the pointer p was changed
   * */
  OffsetPointer ReallocateOffsetNoNullCheck(
    OffsetPointer p, size_t new_size) override;

  /**
   * Free \a ptr pointer. Null check is performed elsewhere.
   * */
  void FreeOffsetNoNullCheck(OffsetPointer p) override;

  /**
   * Get the current amount of data allocated. Can be used for leak
   * checking.
   * */
  size_t GetCurrentlyAllocatedSize() override;

 private:
  /** Round a number up to the nearest page size. */
  size_t RoundUp(size_t num, int &exp);
};

}  // namespace hermes::ipc

#endif  // HERMES_MEMORY_ALLOCATOR_SCALABLE_PAGE_ALLOCATOR_H
