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
#include "hermes_shm/data_structures/ipc/pair.h"
#include "hermes_shm/data_structures/ipc/vector.h"
#include "hermes_shm/data_structures/ipc/list.h"
#include "hermes_shm/data_structures/ipc/pair.h"
#include <hermes_shm/memory/allocator/stack_allocator.h>
#include "mp_page.h"

namespace hshm::ipc {

template<typename AllocT = StackAllocator>
struct ScalablePageAllocatorHeader : public AllocatorHeader {
  hipc::atomic<hshm::min_u64> total_alloc_;
  size_t coalesce_trigger_;
  size_t coalesce_window_;

  HSHM_CROSS_FUN
  ScalablePageAllocatorHeader() = default;

  HSHM_CROSS_FUN
  void Configure(AllocatorId alloc_id,
                 size_t custom_header_size,
                 AllocT *alloc,
                 size_t buffer_size,
                 RealNumber coalesce_trigger,
                 size_t coalesce_window) {
    AllocatorHeader::Configure(alloc_id,
                               AllocatorType::kScalablePageAllocator,
                               custom_header_size);
    total_alloc_ = 0;
    coalesce_trigger_ = (coalesce_trigger * buffer_size).as_int();
    coalesce_window_ = coalesce_window;
  }
};

class ScalablePageAllocator : public Allocator {
 private:
  ScalablePageAllocatorHeader<> *header_;
  StackAllocator alloc_;
  /** The power-of-two exponent of the minimum size that can be cached */
  static const size_t min_cached_size_exp_ = 6;
  /** The minimum size that can be cached directly (64 bytes) */
  static const size_t min_cached_size_ =
    (1 << min_cached_size_exp_) + sizeof(MpPage);
  /** The power-of-two exponent of the minimum size that can be cached (16KB) */
  static const size_t max_cached_size_exp_ = 24;
  /** The maximum size that can be cached directly */
  static const size_t max_cached_size_ =
    (1 << max_cached_size_exp_) + sizeof(MpPage);
  /** The number of well-defined caches */
  static const size_t num_caches_ =
    max_cached_size_exp_ - min_cached_size_exp_ + 1;
  /** An arbitrary free list */
  static const size_t num_free_lists_ = num_caches_ + 1;

 public:
  /**
   * Allocator constructor
   * */
  HSHM_CROSS_FUN
  ScalablePageAllocator()
    : header_(nullptr) {}

  /**
   * Initialize the allocator in shared memory
   * */
  HSHM_CROSS_FUN
  void shm_init(AllocatorId id,
                size_t custom_header_size,
                char *buffer,
                size_t buffer_size,
                RealNumber coalesce_trigger = RealNumber(1, 5),
                size_t coalesce_window = MEGABYTES(1)) {
    type_ = AllocatorType::kScalablePageAllocator;
    id_ = id;
    buffer_ = buffer;
    buffer_size_ = buffer_size;
    header_ = reinterpret_cast<ScalablePageAllocatorHeader<>*>(buffer_);
    custom_header_ = reinterpret_cast<char*>(header_ + 1);
    size_t region_off = (custom_header_ - buffer_) + custom_header_size;
    size_t region_size = buffer_size_ - region_off;
    AllocatorId sub_id(id.bits_.major_, id.bits_.minor_ + 1);
    alloc_.shm_init(sub_id, 0, buffer + region_off, region_size);
    HERMES_MEMORY_MANAGER->RegisterSubAllocator(&alloc_);
    header_->Configure(id, custom_header_size, &alloc_,
                       buffer_size, coalesce_trigger, coalesce_window);
  }

  /**
   * Attach an existing allocator from shared memory
   * */
  HSHM_CROSS_FUN
  void shm_deserialize(char *buffer,
                       size_t buffer_size) override {
    buffer_ = buffer;
    buffer_size_ = buffer_size;
    header_ = reinterpret_cast<ScalablePageAllocatorHeader<>*>(buffer_);
    type_ = header_->allocator_type_;
    id_ = header_->allocator_id_;
    custom_header_ = reinterpret_cast<char*>(header_ + 1);
    size_t region_off = (custom_header_ - buffer_) + header_->custom_header_size_;
    size_t region_size = buffer_size_ - region_off;
    alloc_.shm_deserialize(buffer + region_off, region_size);
    HERMES_MEMORY_MANAGER->RegisterSubAllocator(&alloc_);
  }

  /**
   * Allocate a memory of \a size size. The page allocator cannot allocate
   * memory larger than the page size.
   * */
  HSHM_CROSS_FUN
  OffsetPointer AllocateOffset(const hshm::ThreadId &tid,
                               size_t size) override {
    MpPage *page = nullptr;
    size_t exp;
    size_t size_mp = RoundUp(size + sizeof(MpPage), exp);

    // Case 1: Can we re-use an existing page?
    // page = CheckLocalCaches(size_mp, exp);

    // Case 2: Coalesce if enough space is being wasted
    // if (page == nullptr) {}

    // Case 3: Allocate from stack if no page found
    if (page == nullptr) {
      auto off = alloc_.AllocateOffset(ThreadId::GetNull(), size_mp);
      if (!off.IsNull()) {
        page = alloc_.Convert<MpPage>(off - sizeof(MpPage));
      }
    }

    // Case 4: Completely out of memory
    if (page == nullptr) {
      HERMES_THROW_ERROR(OUT_OF_MEMORY, size,
                         GetCurrentlyAllocatedSize());
    }

    // Mark as allocated
    page->page_size_ = size_mp;
    header_->total_alloc_.fetch_add(page->page_size_);
    auto p = Convert<MpPage, OffsetPointer>(page);
    page->SetAllocated();
    return p + sizeof(MpPage);
  }

 public:
  /**
   * Allocate a memory of \a size size, which is aligned to \a
   * alignment.
   * */
  HSHM_CROSS_FUN
  OffsetPointer AlignedAllocateOffset(const hshm::ThreadId &tid,
                                      size_t size,
                                      size_t alignment) override {
    HERMES_THROW_ERROR(NOT_IMPLEMENTED, "AlignedAllocateOffset");
  }

  /**
   * Reallocate \a p pointer to \a new_size new size.
   *
   * @return whether or not the pointer p was changed
   * */
  HSHM_CROSS_FUN
  OffsetPointer ReallocateOffsetNoNullCheck(
      const hshm::ThreadId &tid,
      OffsetPointer p, size_t new_size) override {
    OffsetPointer new_p;
    void *ptr = AllocatePtr<void*, OffsetPointer>(tid, new_size, new_p);
    MpPage *hdr = Convert<MpPage>(p - sizeof(MpPage));
    void *old = (void*)(hdr + 1);
    memcpy(ptr, old, hdr->page_size_ - sizeof(MpPage));
    FreeOffsetNoNullCheck(ThreadId::GetNull(), p);
    return new_p;
  }

  /**
   * Free \a ptr pointer. Null check is performed elsewhere.
   * */
  HSHM_CROSS_FUN
  void FreeOffsetNoNullCheck(const hshm::ThreadId &tid,
                             OffsetPointer p) override {
    // Mark as free
    auto hdr_offset = p - sizeof(MpPage);
    auto hdr = Convert<MpPage>(hdr_offset);
    if (!hdr->IsAllocated()) {
      HERMES_THROW_ERROR(DOUBLE_FREE);
    }
    hdr->UnsetAllocated();
    header_->total_alloc_.fetch_sub(hdr->page_size_);
    size_t exp;
    RoundUp(hdr->page_size_, exp);
  }

  /**
   * Get the current amount of data allocated. Can be used for leak
   * checking.
   * */
  HSHM_CROSS_FUN
  size_t GetCurrentlyAllocatedSize() override {
    return header_->total_alloc_.load();
  }

 private:
  /** Round a number up to the nearest page size. */
  HSHM_INLINE_CROSS_FUN size_t RoundUp(size_t num, size_t &exp) {
    size_t round;
    for (exp = 0; exp < num_caches_; ++exp) {
      round = 1 << (exp + min_cached_size_exp_);
      round += sizeof(MpPage);
      if (num <= round) {
        return round;
      }
    }
    return num;
  }
};

}  // namespace hshm::ipc

#endif  // HERMES_MEMORY_ALLOCATOR_SCALABLE_PAGE_ALLOCATOR_H
