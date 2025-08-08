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

#ifndef HSHM_MEMORY_ALLOCATOR_STACK_ALLOCATOR_H_
#define HSHM_MEMORY_ALLOCATOR_STACK_ALLOCATOR_H_

#include "allocator.h"
#include "heap.h"
#include "hermes_shm/memory/allocator/mp_page.h"
#include "hermes_shm/thread/lock.h"

namespace hshm::ipc {

class _StackAllocator;
typedef BaseAllocator<_StackAllocator> StackAllocator;

struct _StackAllocatorHeader : public AllocatorHeader {
  HeapAllocator<true> heap_;
  hipc::atomic<hshm::size_t> total_alloc_;

  HSHM_CROSS_FUN
  _StackAllocatorHeader() = default;

  HSHM_CROSS_FUN
  void Configure(AllocatorId alloc_id, size_t custom_header_size,
                 size_t region_off, size_t region_size) {
    AllocatorHeader::Configure(alloc_id, AllocatorType::kStackAllocator,
                               custom_header_size);
    heap_.shm_init(region_off, region_size);
    total_alloc_ = 0;
  }
};

class _StackAllocator : public Allocator {
 public:
  HSHM_ALLOCATOR(_StackAllocator);

 public:
  typedef BaseAllocator<_StackAllocator> AllocT;
  _StackAllocatorHeader *header_;
  HeapAllocator<true> *heap_;

 public:
  /**
   * Allocator constructor
   * */
  HSHM_CROSS_FUN
  _StackAllocator() : header_(nullptr) {}

  /**
   * Initialize the allocator in shared memory
   * */
  HSHM_CROSS_FUN
  void shm_init(AllocatorId id, size_t custom_header_size,
                MemoryBackend backend) {
    type_ = AllocatorType::kStackAllocator;
    id_ = id;
    buffer_ = backend.data_;
    buffer_size_ = backend.data_size_;
    header_ = ConstructHeader<_StackAllocatorHeader>(buffer_);
    custom_header_ = reinterpret_cast<char *>(header_ + 1);
    size_t region_off = (custom_header_ - buffer_) + custom_header_size;
    size_t region_size = buffer_size_ - region_off;
    header_->Configure(id, custom_header_size, region_off, region_size);
    heap_ = &header_->heap_;
  }

  /**
   * Attach an existing allocator from shared memory
   * */
  HSHM_CROSS_FUN
  void shm_deserialize(MemoryBackend backend) {
    buffer_ = backend.data_;
    buffer_size_ = backend.data_size_;
    header_ = reinterpret_cast<_StackAllocatorHeader *>(buffer_);
    type_ = header_->allocator_type_;
    id_ = header_->alloc_id_;
    custom_header_ = reinterpret_cast<char *>(header_ + 1);
    heap_ = &header_->heap_;
  }

  /**
   * Allocate a memory of \a size size. The page allocator cannot allocate
   * memory larger than the page size.
   * */
  HSHM_INLINE_CROSS_FUN
  OffsetPointer SubAllocateOffset(size_t size) {
    return heap_->AllocateOffset(size);
  }

  /** Align the memory to the next page boundary */
  HSHM_CROSS_FUN
  void Align() {
    hshm::size_t off = heap_->heap_off_.load();
    hshm::size_t page_size = 4096;
    hshm::size_t new_off = (off + page_size - 1) & ~(page_size - 1);
    heap_->heap_off_.store(new_off);
  }

  /**
   * Allocate a memory of \a size size. The page allocator cannot allocate
   * memory larger than the page size.
   * */
  HSHM_CROSS_FUN
  OffsetPointer AllocateOffset(const hipc::MemContext &ctx, size_t size) {
    size += sizeof(MpPage);
    OffsetPointer p = heap_->AllocateOffset(size);
    auto hdr = Convert<MpPage>(p);
    hdr->SetAllocated();
    hdr->page_size_ = size;
    hdr->off_ = 0;
    header_->AddSize(hdr->page_size_);
    return p + sizeof(MpPage);
  }

  /**
   * Allocate a memory of \a size size, which is aligned to \a
   * alignment.
   * */
  HSHM_CROSS_FUN
  OffsetPointer AlignedAllocateOffset(const hipc::MemContext &ctx, size_t size,
                                      size_t alignment) {
    HSHM_THROW_ERROR(NOT_IMPLEMENTED, "AlignedAllocateOffset");
    return OffsetPointer::GetNull();
  }

  /**
   * Reallocate \a p pointer to \a new_size new size.
   *
   * @return whether or not the pointer p was changed
   * */
  HSHM_CROSS_FUN
  OffsetPointer ReallocateOffsetNoNullCheck(const hipc::MemContext &ctx,
                                            OffsetPointer p, size_t new_size) {
    OffsetPointer new_p;
    void *src = Convert<void>(p);
    auto hdr = Convert<MpPage>(p - sizeof(MpPage));
    size_t old_size = hdr->page_size_ - sizeof(MpPage);
    auto full_ptr = ((AllocT *)this)->template Allocate<void, OffsetPointer>(ctx, new_size);
    new_p = full_ptr.shm_;
    void *dst = full_ptr.ptr_;
    memcpy((void *)dst, (void *)src, old_size);
    FreeOffsetNoNullCheck(ctx, p);
    return new_p;
  }

  /**
   * Free \a ptr pointer. Null check is performed elsewhere.
   * */
  HSHM_CROSS_FUN
  void FreeOffsetNoNullCheck(const hipc::MemContext &ctx, OffsetPointer p) {
    auto hdr = Convert<MpPage>(p - sizeof(MpPage));
    if (!hdr->IsAllocated()) {
      HSHM_THROW_ERROR(DOUBLE_FREE);
    }
    hdr->UnsetAllocated();
    header_->SubSize(hdr->page_size_);
  }

  /**
   * Get the current amount of data allocated. Can be used for leak
   * checking.
   * */
  HSHM_CROSS_FUN
  size_t GetCurrentlyAllocatedSize() {
    return (size_t)header_->GetCurrentlyAllocatedSize();
  }

  /**
   * Create a globally-unique thread ID
   * */
  HSHM_CROSS_FUN
  void CreateTls(MemContext &ctx) {}

  /**
   * Free a thread-local memory storage
   * */
  HSHM_CROSS_FUN
  void FreeTls(const hipc::MemContext &ctx) {}
};

}  // namespace hshm::ipc

#endif  // HSHM_MEMORY_ALLOCATOR_STACK_ALLOCATOR_H_
