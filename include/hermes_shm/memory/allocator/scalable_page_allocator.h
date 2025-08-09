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

#ifndef HSHM_MEMORY_ALLOCATOR_SCALABLE_PAGE_ALLOCATOR_H
#define HSHM_MEMORY_ALLOCATOR_SCALABLE_PAGE_ALLOCATOR_H

#include <cmath>

#include "allocator.h"
#include "hermes_shm/data_structures/ipc/list.h"
#include "hermes_shm/data_structures/ipc/pair.h"
#include "hermes_shm/data_structures/ipc/ring_ptr_queue.h"
#include "hermes_shm/data_structures/ipc/ring_queue.h"
#include "hermes_shm/data_structures/ipc/vector.h"
#include "hermes_shm/memory/allocator/stack_allocator.h"
#include "hermes_shm/thread/lock.h"
#include "hermes_shm/util/timer.h"
#include "mp_page.h"
#include "page_allocator.h"

namespace hshm::ipc {

class _ScalablePageAllocator;
typedef BaseAllocator<_ScalablePageAllocator> ScalablePageAllocator;

struct _ScalablePageAllocatorHeader : public AllocatorHeader {
  typedef hipc::PageAllocator<_ScalablePageAllocator, true, false>
      PageAllocator;
  hipc::atomic<hshm::size_t> total_alloc_;
  hipc::delay_ar<PageAllocator> global_;

  HSHM_CROSS_FUN
  _ScalablePageAllocatorHeader() = default;

  HSHM_CROSS_FUN
  void Configure(AllocatorId alloc_id, size_t custom_header_size,
                 StackAllocator *alloc) {
    AllocatorHeader::Configure(alloc_id, AllocatorType::kScalablePageAllocator,
                               custom_header_size);
    total_alloc_ = 0;
    // TODO: Fix this specific case - PageAllocator needs special handling
    // global_.shm_init(alloc, hshm::Unit<size_t>::Kilobytes(1));
    Allocator::ConstructObj(global_.get_ref(), alloc, hshm::Unit<size_t>::Kilobytes(1));
  }
};

class _ScalablePageAllocator : public Allocator {
 public:
  HSHM_ALLOCATOR(_ScalablePageAllocator);

 private:
  typedef _ScalablePageAllocatorHeader::PageAllocator PageAllocator;
  _ScalablePageAllocatorHeader *header_;
  StackAllocator alloc_;

 public:
  /**
   * Allocator constructor
   * */
  HSHM_CROSS_FUN
  _ScalablePageAllocator() : header_(nullptr) {}

  /**
   * Initialize the allocator in shared memory
   * */
  HSHM_CROSS_FUN
  void shm_init(AllocatorId id, size_t custom_header_size,
                MemoryBackend backend) {
    type_ = AllocatorType::kScalablePageAllocator;
    id_ = id;
    buffer_ = backend.data_;
    buffer_size_ = backend.data_size_;
    header_ = ConstructHeader<_ScalablePageAllocatorHeader>(buffer_);
    custom_header_ = reinterpret_cast<char *>(header_ + 1);
    size_t region_off = (custom_header_ - buffer_) + custom_header_size;
    size_t region_size = buffer_size_ - region_off;
    AllocatorId sub_id(id.bits_.major_, id.bits_.minor_ + 1);
    alloc_.shm_init(sub_id, 0, backend.Shift(region_off));
    HSHM_MEMORY_MANAGER->RegisterSubAllocator(&alloc_);
    header_->Configure(id, custom_header_size, &alloc_);
    alloc_.Align();
  }

  /**
   * Attach an existing allocator from shared memory
   * */
  HSHM_CROSS_FUN
  void shm_deserialize(MemoryBackend backend) {
    buffer_ = backend.data_;
    buffer_size_ = backend.data_size_;
    header_ = reinterpret_cast<_ScalablePageAllocatorHeader *>(buffer_);
    type_ = header_->allocator_type_;
    id_ = header_->alloc_id_;
    custom_header_ = reinterpret_cast<char *>(header_ + 1);
    size_t region_off =
        (custom_header_ - buffer_) + header_->custom_header_size_;
    size_t region_size = buffer_size_ - region_off;
    alloc_.shm_deserialize(backend.Shift(region_off));
    HSHM_MEMORY_MANAGER->RegisterSubAllocator(&alloc_);
  }

  /**
   * Allocate a memory of \a size size. The page allocator cannot allocate
   * memory larger than the page size.
   * */
  HSHM_CROSS_FUN
  OffsetPointer AllocateOffset(const hipc::MemContext &ctx, size_t size) {
    MpPage *page = nullptr;
    PageId page_id(size + sizeof(MpPage));

    // Case 1: Can we re-use an existing page?
    PageAllocator &page_alloc = *header_->global_;
    page = page_alloc.Allocate(page_id);

    // Case 2: Coalesce if enough space is being wasted
    // if (page == nullptr) {}

    // Case 3: Allocate from heap if no page found
    if (page == nullptr) {
      OffsetPointer off = alloc_.SubAllocateOffset(page_id.round_);
      if (!off.IsNull()) {
        page = alloc_.Convert<MpPage>(off);
      }
    }

    // Case 4: Completely out of memory
    if (page == nullptr) {
      HSHM_THROW_ERROR(OUT_OF_MEMORY, size, GetCurrentlyAllocatedSize());
    }

    // Mark as allocated
    header_->AddSize(page_id.round_);
    OffsetPointer p = Convert<MpPage, OffsetPointer>(page);
    page->page_size_ = page_id.round_;
    page->SetAllocated();
    return p + sizeof(MpPage);
  }

 public:
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
    auto full_ptr = GetAllocator()->template Allocate<void, OffsetPointer>(ctx, new_size);
    FullPtr<char, OffsetPointer> new_ptr(reinterpret_cast<char*>(full_ptr.ptr_), full_ptr.shm_);
    char *old = Convert<char, OffsetPointer>(p);
    MpPage *old_hdr = (MpPage *)(old - sizeof(MpPage));
    memcpy(new_ptr.ptr_, old, old_hdr->page_size_ - sizeof(MpPage));
    FreeOffsetNoNullCheck(ctx.tid_, p);
    return new_ptr.shm_;
  }

  /**
   * Free \a ptr pointer. Null check is performed elsewhere.
   * */
  HSHM_CROSS_FUN
  void FreeOffsetNoNullCheck(const hipc::MemContext &ctx, OffsetPointer p) {
    // Mark as free
    auto hdr_offset = p - sizeof(MpPage);
    MpPage *hdr = Convert<MpPage>(hdr_offset);
    if (!hdr->IsAllocated()) {
      HSHM_THROW_ERROR(DOUBLE_FREE, hdr);
    }
    hdr->UnsetAllocated();
    header_->SubSize(hdr->page_size_);
    PageAllocator &page_alloc = *header_->global_;
    page_alloc.Free(hdr_offset, hdr);
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
  void FreeTls(const MemContext &ctx) {}
};

}  // namespace hshm::ipc

#endif  // HSHM_MEMORY_ALLOCATOR_SCALABLE_PAGE_ALLOCATOR_H
