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


#ifndef HERMES_MEMORY_ALLOCATOR_STACK_ALLOCATOR_H_
#define HERMES_MEMORY_ALLOCATOR_STACK_ALLOCATOR_H_

#include "allocator.h"
#include "heap.h"
#include "hermes_shm/thread/lock.h"
#include <hermes_shm/memory/allocator/mp_page.h>

namespace hshm::ipc {

struct StackAllocatorHeader : public AllocatorHeader {
  HeapAllocator heap_;
  std::atomic<size_t> total_alloc_;

  StackAllocatorHeader() = default;

  void Configure(allocator_id_t alloc_id,
                 size_t custom_header_size,
                 size_t region_off,
                 size_t region_size) {
    AllocatorHeader::Configure(alloc_id, AllocatorType::kStackAllocator,
                               custom_header_size);
    heap_.shm_init(region_off, region_size);
    total_alloc_ = 0;
  }
};

class StackAllocator : public Allocator {
 public:
  StackAllocatorHeader *header_;
  HeapAllocator *heap_;

 public:
  /**
   * Allocator constructor
   * */
  StackAllocator()
  : header_(nullptr) {}

  /**
   * Get the ID of this allocator from shared memory
   * */
  allocator_id_t &GetId() override {
    return header_->allocator_id_;
  }

  /**
   * Initialize the allocator in shared memory
   * */
  void shm_init(allocator_id_t id,
                size_t custom_header_size,
                char *buffer,
                size_t buffer_size) {
    buffer_ = buffer;
    buffer_size_ = buffer_size;
    header_ = reinterpret_cast<StackAllocatorHeader*>(buffer_);
    custom_header_ = reinterpret_cast<char*>(header_ + 1);
    size_t region_off = (custom_header_ - buffer_) + custom_header_size;
    size_t region_size = buffer_size_ - region_off;
    header_->Configure(id, custom_header_size, region_off, region_size);
    heap_ = &header_->heap_;
  }

  /**
   * Attach an existing allocator from shared memory
   * */
  void shm_deserialize(char *buffer,
                       size_t buffer_size) override {
    buffer_ = buffer;
    buffer_size_ = buffer_size;
    header_ = reinterpret_cast<StackAllocatorHeader*>(buffer_);
    custom_header_ = reinterpret_cast<char*>(header_ + 1);
    heap_ = &header_->heap_;
  }

  /**
   * Allocate a memory of \a size size. The page allocator cannot allocate
   * memory larger than the page size.
   * */
  OffsetPointer AllocateOffset(size_t size) override {
    size += sizeof(MpPage);
    OffsetPointer p = heap_->AllocateOffset(size);
    auto hdr = Convert<MpPage>(p);
    hdr->SetAllocated();
    hdr->page_size_ = size;
    hdr->off_ = 0;
    header_->total_alloc_.fetch_add(hdr->page_size_);
    return p + sizeof(MpPage);
  }

  /**
   * Allocate a memory of \a size size, which is aligned to \a
   * alignment.
   * */
  OffsetPointer AlignedAllocateOffset(size_t size, size_t alignment) override {
    throw ALIGNED_ALLOC_NOT_SUPPORTED.format();
  }

  /**
   * Reallocate \a p pointer to \a new_size new size.
   *
   * @return whether or not the pointer p was changed
   * */
  OffsetPointer ReallocateOffsetNoNullCheck(
    OffsetPointer p, size_t new_size) override {
    OffsetPointer new_p;
    void *src = Convert<void>(p);
    auto hdr = Convert<MpPage>(p - sizeof(MpPage));
    size_t old_size = hdr->page_size_ - sizeof(MpPage);
    void *dst = AllocatePtr<void, OffsetPointer>(new_size, new_p);
    memcpy((void*)dst, (void*)src, old_size);
    Free(p);
    return new_p;
  }

  /**
   * Free \a ptr pointer. Null check is performed elsewhere.
   * */
  void FreeOffsetNoNullCheck(OffsetPointer p) override {
    auto hdr = Convert<MpPage>(p - sizeof(MpPage));
    if (!hdr->IsAllocated()) {
      throw DOUBLE_FREE.format();
    }
    hdr->UnsetAllocated();
    header_->total_alloc_.fetch_sub(hdr->page_size_);
  }

  /**
   * Get the current amount of data allocated. Can be used for leak
   * checking.
   * */
  size_t GetCurrentlyAllocatedSize() override {
    return header_->total_alloc_;
  }
};

}  // namespace hshm::ipc

#endif  // HERMES_MEMORY_ALLOCATOR_STACK_ALLOCATOR_H_
