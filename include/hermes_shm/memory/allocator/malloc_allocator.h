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


#ifndef HERMES_MEMORY_ALLOCATOR_MALLOC_ALLOCATOR_H_
#define HERMES_MEMORY_ALLOCATOR_MALLOC_ALLOCATOR_H_

#include "allocator.h"
#include "hermes_shm/thread/lock.h"

namespace hshm::ipc {

struct MallocPage {
  size_t page_size_;
};

struct _MallocAllocatorHeader : public AllocatorHeader {
  hipc::atomic<hshm::min_u64> total_alloc_size_;

  HSHM_CROSS_FUN
  _MallocAllocatorHeader() = default;

  HSHM_CROSS_FUN
  void Configure(AllocatorId alloc_id,
                 size_t custom_header_size) {
    AllocatorHeader::Configure(alloc_id, AllocatorType::kStackAllocator,
                               custom_header_size);
    total_alloc_size_ = 0;
  }
};

class _MallocAllocator : public Allocator {
 private:
  _MallocAllocatorHeader *header_;

 public:
  /**
   * Allocator constructor
   * */
  HSHM_CROSS_FUN
  _MallocAllocator()
  : header_(nullptr) {}

  /**
   * Initialize the allocator in shared memory
   * */
  HSHM_CROSS_FUN
  void shm_init(AllocatorId id,
                size_t custom_header_size,
                size_t buffer_size)  {
    type_ = AllocatorType::kMallocAllocator;
    id_ = id;
    buffer_ = nullptr;
    buffer_size_ = buffer_size;
    header_ = reinterpret_cast<_MallocAllocatorHeader*>(
        malloc(sizeof(_MallocAllocatorHeader) + custom_header_size));
    custom_header_ = reinterpret_cast<char*>(header_ + 1);
    header_->Configure(id, custom_header_size);
  }

  /**
   * Attach an existing allocator from shared memory
   * */
  HSHM_CROSS_FUN
  void shm_deserialize(char *buffer,
                       size_t buffer_size)  {
    HERMES_THROW_ERROR(NOT_IMPLEMENTED, "_MallocAllocator::shm_deserialize");
  }

  /**
   * Allocate a memory of \a size size. The page allocator cannot allocate
   * memory larger than the page size.
   * */
  HSHM_CROSS_FUN
  OffsetPointer AllocateOffset(const hipc::MemContext &ctx,
                               size_t size) {
    auto page = reinterpret_cast<MallocPage*>(
        malloc(sizeof(MallocPage) + size));
    page->page_size_ = size;
    header_->total_alloc_size_ += size;
    return OffsetPointer((size_t)(page + 1));
  }

  /**
   * Allocate a memory of \a size size, which is aligned to \a
   * alignment.
   * */
  HSHM_CROSS_FUN
  OffsetPointer AlignedAllocateOffset(const hipc::MemContext &ctx,
                                      size_t size, size_t alignment) {
#ifdef HSHM_IS_HOST
    auto page = reinterpret_cast<MallocPage*>(
        aligned_alloc(alignment, sizeof(MallocPage) + size));
    page->page_size_ = size;
    header_->total_alloc_size_ += size;
    return OffsetPointer(size_t(page + 1));
#else
    return OffsetPointer(0);
#endif
  }

  /**
   * Reallocate \a p pointer to \a new_size new size.
   *
   * @return whether or not the pointer p was changed
   * */
  HSHM_CROSS_FUN
  OffsetPointer ReallocateOffsetNoNullCheck(const hipc::MemContext &ctx,
                                            OffsetPointer p,
                                            size_t new_size) {
#ifdef HSHM_IS_HOST
    // Get the input page
    auto page = reinterpret_cast<MallocPage*>(
        p.off_.load() - sizeof(MallocPage));
    header_->total_alloc_size_ += new_size - page->page_size_;

    // Reallocate the input page
    auto new_page = reinterpret_cast<MallocPage*>(
        realloc(page, sizeof(MallocPage) + new_size));
    new_page->page_size_ = new_size;

    // Create the pointer
    return OffsetPointer(size_t(new_page + 1));
#else
    return OffsetPointer(0);
#endif
  }

  /**
   * Free \a ptr pointer. Null check is performed elsewhere.
   * */
  HSHM_CROSS_FUN
  void FreeOffsetNoNullCheck(const hipc::MemContext &ctx,
                             OffsetPointer p) {
    auto page = reinterpret_cast<MallocPage*>(
        p.off_.load() - sizeof(MallocPage));
    header_->total_alloc_size_ -= page->page_size_;
    free(page);
  }

  /**
   * Get the current amount of data allocated. Can be used for leak
   * checking.
   * */
  HSHM_CROSS_FUN
  size_t GetCurrentlyAllocatedSize() {
    return header_->total_alloc_size_.load();
  }

  /**
   * Create a globally-unique thread ID
   * */
  HSHM_CROSS_FUN
  void CreateTls(MemContext &ctx) {
  }

  /**
   * Free a thread-local memory storage
   * */
  HSHM_CROSS_FUN
  void FreeTls(const hipc::MemContext &ctx) {
  }
};

typedef BaseAllocator<_MallocAllocator> MallocAllocator;

}  // namespace hshm::ipc

#endif  // HERMES_MEMORY_ALLOCATOR_MALLOC_ALLOCATOR_H_
