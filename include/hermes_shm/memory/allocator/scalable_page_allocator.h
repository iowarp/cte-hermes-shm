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
#include "hermes_shm/data_structures/ipc/ring_queue.h"
#include <hermes_shm/memory/allocator/stack_allocator.h>
#include <cmath>
#include "mp_page.h"

namespace hshm::ipc {

struct PageId {
 public:
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
  size_t orig_;
  size_t round_;
  size_t exp_;

 public:
  /**
   * Round the size of the requested memory region + sizeof(MpPage)
   * to the nearest power of two.
   * */
  HSHM_INLINE_CROSS_FUN
  PageId(size_t size) {
    orig_ = size;
#ifndef __CUDA_ARCH__
    exp_ = std::ceil(std::log2(size - sizeof(MpPage)));
#else
    exp_ = ceil(log2(num));
#endif
    round_ = (1 << exp_) + sizeof(MpPage);
    if (exp_ < min_cached_size_exp_) {
      round_ = min_cached_size_;
      exp_ = min_cached_size_exp_;
    } else if (exp_ > max_cached_size_exp_) {
      round_ = size;
      exp_ = max_cached_size_exp_;
    } else {
      round_ = (1 << exp_) + sizeof(MpPage);
      exp_ -= min_cached_size_exp_;
    }
  }
};

class PageAllocator {
 public:
  hipc::delay_ar<hipc::iqueue<MpPage, StackAllocator>>
    free_lists_[PageId::num_free_lists_];

 public:
  explicit PageAllocator(StackAllocator *alloc) {
    for (size_t i = 0; i < PageId::num_free_lists_; ++i) {
      HSHM_MAKE_AR0(free_lists_[i], alloc)
    }
  }

  PageAllocator(const PageAllocator &other) {}
  PageAllocator(PageAllocator &&other) {}

  MpPage* Allocate(const PageId &page_id) {
    // Allocate small page size
    if (page_id.exp_ < PageId::num_caches_) {
      return free_lists_[page_id.exp_]->dequeue();
    }
    // Allocate a large page size
    for (auto it = free_lists_[PageId::num_caches_]->begin();
         it != free_lists_[PageId::num_caches_]->end(); ++it) {
      MpPage *page = *it;
      if (page->page_size_ >= page_id.round_) {
        free_lists_[PageId::num_caches_]->dequeue(it);
        return page;
      }
    }
    // No page was cached
    return nullptr;
  }

  void Free(MpPage *page) {
    PageId page_id(page->page_size_);
    if (page_id.exp_ < PageId::num_caches_) {
      free_lists_[page_id.exp_]->enqueue(page);
    } else {
      free_lists_[PageId::num_caches_]->enqueue(page);
    }
  }
};

struct ScalablePageAllocatorHeader : public AllocatorHeader {
  hipc::delay_ar<hipc::vector<PageAllocator>> tls_;
  hipc::atomic<hshm::min_u64> total_alloc_;
  size_t coalesce_trigger_;
  size_t coalesce_window_;

  HSHM_CROSS_FUN
  ScalablePageAllocatorHeader() = default;

  HSHM_CROSS_FUN
  void Configure(AllocatorId alloc_id,
                 size_t custom_header_size,
                 StackAllocator *alloc,
                 size_t buffer_size,
                 RealNumber coalesce_trigger,
                 size_t coalesce_window) {
    AllocatorHeader::Configure(alloc_id,
                               AllocatorType::kScalablePageAllocator,
                               custom_header_size);
    HSHM_MAKE_AR(tls_, alloc, (1<<20), alloc);
    total_alloc_ = 0;
    coalesce_trigger_ = (coalesce_trigger * buffer_size).as_int();
    coalesce_window_ = coalesce_window;
  }
};

class ScalablePageAllocator : public Allocator {
 private:
  ScalablePageAllocatorHeader *header_;
  StackAllocator alloc_;

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
    header_ = reinterpret_cast<ScalablePageAllocatorHeader*>(buffer_);
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
    header_ = reinterpret_cast<ScalablePageAllocatorHeader*>(buffer_);
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
  OffsetPointer AllocateOffset(hshm::ThreadId tid,
                               size_t size) override {
    PageId page_id(size + sizeof(MpPage));

    // Case 1: Can we re-use an existing page?
    if (tid.IsNull()) {
      tid = HERMES_THREAD_MODEL->GetTid();
    }
    PageAllocator &page_alloc = (*header_->tls_)[tid.tid_];
    MpPage *page = page_alloc.Allocate(page_id);

    // Case 2: Coalesce if enough space is being wasted
    // if (page == nullptr) {}

    // Case 3: Allocate from heap if no page found
    if (page == nullptr) {
      auto off = alloc_.AllocateOffset(tid, page_id.round_);
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
    page->page_size_ = page_id.round_;
    header_->total_alloc_.fetch_add(page->page_size_);
    OffsetPointer p = Convert<MpPage, OffsetPointer>(page);
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
    LPointer<char, OffsetPointer> new_ptr =
        AllocateLocalPtr<char, OffsetPointer>(tid, new_size);
    char *old = Convert<char, OffsetPointer>(p);
    MpPage *old_hdr = (MpPage*)(old - sizeof(MpPage));
    memcpy(new_ptr.ptr_, old, old_hdr->page_size_ - sizeof(MpPage));
    FreeOffsetNoNullCheck(tid, p);
    return new_ptr.shm_;
  }

  /**
   * Free \a ptr pointer. Null check is performed elsewhere.
   * */
  HSHM_CROSS_FUN
  void FreeOffsetNoNullCheck(hshm::ThreadId tid,
                             OffsetPointer p) override {
    // Mark as free
    auto hdr_offset = p - sizeof(MpPage);
    MpPage *hdr = Convert<MpPage>(hdr_offset);
    if (!hdr->IsAllocated()) {
      HERMES_THROW_ERROR(DOUBLE_FREE);
    }
    hdr->UnsetAllocated();
    header_->total_alloc_.fetch_sub(hdr->page_size_);
    if (tid.IsNull()) {
      tid = HERMES_THREAD_MODEL->GetTid();
    }
    PageAllocator &page_alloc = (*header_->tls_)[tid.tid_];
    page_alloc.Free(hdr);
  }

  /**
   * Get the current amount of data allocated. Can be used for leak
   * checking.
   * */
  HSHM_CROSS_FUN
  size_t GetCurrentlyAllocatedSize() override {
    return header_->total_alloc_.load();
  }

  /**
   * Free a thread-local memory storage
   * */
  HSHM_CROSS_FUN
  void FreeTls(ThreadId tid) override {
    if (tid.IsNull()) {
      tid = HERMES_THREAD_MODEL->GetTid();
    }
  }

 private:

};

}  // namespace hshm::ipc

#endif  // HERMES_MEMORY_ALLOCATOR_SCALABLE_PAGE_ALLOCATOR_H
