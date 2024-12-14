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

#include <hermes_shm/memory/allocator/stack_allocator.h>

#include <cmath>

#include "allocator.h"
#include "hermes_shm/data_structures/ipc/list.h"
#include "hermes_shm/data_structures/ipc/pair.h"
#include "hermes_shm/data_structures/ipc/ring_ptr_queue.h"
#include "hermes_shm/data_structures/ipc/ring_queue.h"
#include "hermes_shm/data_structures/ipc/vector.h"
#include "hermes_shm/thread/lock.h"
#include "hermes_shm/util/timer.h"
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
#ifdef HSHM_IS_HOST
    exp_ = std::ceil(std::log2(size - sizeof(MpPage)));
#else
    exp_ = ceil(log2(size - sizeof(MpPage)));
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

class _ScalablePageAllocator;

class PageAllocator {
 public:
  typedef StackAllocator Alloc_;
  typedef TlsAllocatorInfo<_ScalablePageAllocator> TLS;
  typedef hipc::iqueue<MpPage, Alloc_> LIST;

 public:
  hipc::delay_ar<LIST> free_lists_[PageId::num_free_lists_];
  TLS tls_info_;

 public:
  HSHM_INLINE_CROSS_FUN
  explicit PageAllocator(StackAllocator *alloc) {
    for (size_t i = 0; i < PageId::num_free_lists_; ++i) {
      HSHM_MAKE_AR0(free_lists_[i], alloc);
    }
  }

  HSHM_INLINE_CROSS_FUN
  PageAllocator(const PageAllocator &other) {}

  HSHM_INLINE_CROSS_FUN
  PageAllocator(PageAllocator &&other) {}

  HSHM_INLINE_CROSS_FUN
  MpPage *Allocate(const PageId &page_id) {
    // Allocate small page size
    if (page_id.exp_ < PageId::num_caches_) {
      LIST &free_list = *free_lists_[page_id.exp_];
      MpPage *page = free_list.pop();
      return page;
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

  HSHM_INLINE_CROSS_FUN
  void Free(MpPage *page) {
    PageId page_id(page->page_size_);
    if (page_id.exp_ < PageId::num_caches_) {
      free_lists_[page_id.exp_]->enqueue(page);
    } else {
      free_lists_[PageId::num_caches_]->enqueue(page);
    }
  }
};

struct _ScalablePageAllocatorHeader : public AllocatorHeader {
  typedef TlsAllocatorInfo<_ScalablePageAllocator> TLS;
  typedef hipc::vector<PageAllocator, StackAllocator> PageAllocVec;
  typedef hipc::fixed_mpmc_ptr_queue<hshm::min_u64, StackAllocator>
      PageAllocIdVec;

  hipc::delay_ar<PageAllocVec> tls_;
  hipc::delay_ar<PageAllocIdVec> free_tids_;
  hipc::atomic<hshm::min_u64> tid_heap_;
  hipc::atomic<hshm::min_u64> total_alloc_;

  HSHM_CROSS_FUN
  _ScalablePageAllocatorHeader() = default;

  HSHM_CROSS_FUN
  void Configure(AllocatorId alloc_id, size_t custom_header_size,
                 StackAllocator *alloc, size_t buffer_size,
                 size_t max_threads) {
    AllocatorHeader::Configure(alloc_id, AllocatorType::kScalablePageAllocator,
                               custom_header_size);
    HSHM_MAKE_AR(tls_, alloc, max_threads, alloc);
    HSHM_MAKE_AR(free_tids_, alloc, max_threads);
    total_alloc_ = 0;
  }

  HSHM_INLINE_CROSS_FUN
  hshm::ThreadId CreateTid() {
    hshm::min_u64 tid = 0;
    if (free_tids_->pop(tid).IsNull()) {
      tid = tid_heap_.fetch_add(1);
    }
    return hshm::ThreadId(tid);
  }

  HSHM_INLINE_CROSS_FUN
  void FreeTid(hshm::ThreadId tid) { free_tids_->emplace(tid.tid_); }

  HSHM_INLINE_CROSS_FUN
  TLS *GetTls(hshm::ThreadId tid) { return &(*tls_)[tid.tid_].tls_info_; }
};

class _ScalablePageAllocator : public Allocator {
 private:
  typedef TlsAllocatorInfo<_ScalablePageAllocator> TLS;
  typedef BaseAllocator<_ScalablePageAllocator> AllocT;
  _ScalablePageAllocatorHeader *header_;
  StackAllocator alloc_;
  thread::ThreadLocalKey tls_key_;

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
  void shm_init(AllocatorId id, size_t custom_header_size, char *buffer,
                size_t buffer_size, size_t max_threads = 1024) {
    type_ = AllocatorType::kScalablePageAllocator;
    id_ = id;
    buffer_ = buffer;
    buffer_size_ = buffer_size;
    header_ = reinterpret_cast<_ScalablePageAllocatorHeader *>(buffer_);
    custom_header_ = reinterpret_cast<char *>(header_ + 1);
    size_t region_off = (custom_header_ - buffer_) + custom_header_size;
    size_t region_size = buffer_size_ - region_off;
    AllocatorId sub_id(id.bits_.major_, id.bits_.minor_ + 1);
    alloc_.shm_init(sub_id, 0, buffer + region_off, region_size);
    HERMES_MEMORY_MANAGER->RegisterSubAllocator(&alloc_);
    header_->Configure(id, custom_header_size, &alloc_, buffer_size,
                       max_threads);
    HERMES_THREAD_MODEL->CreateTls<TLS>(tls_key_, nullptr);
    alloc_.Align();
  }

  /**
   * Attach an existing allocator from shared memory
   * */
  HSHM_CROSS_FUN
  void shm_deserialize(char *buffer, size_t buffer_size) {
    buffer_ = buffer;
    buffer_size_ = buffer_size;
    header_ = reinterpret_cast<_ScalablePageAllocatorHeader *>(buffer_);
    type_ = header_->allocator_type_;
    id_ = header_->allocator_id_;
    custom_header_ = reinterpret_cast<char *>(header_ + 1);
    size_t region_off =
        (custom_header_ - buffer_) + header_->custom_header_size_;
    size_t region_size = buffer_size_ - region_off;
    alloc_.shm_deserialize(buffer + region_off, region_size);
    HERMES_MEMORY_MANAGER->RegisterSubAllocator(&alloc_);
    HERMES_THREAD_MODEL->CreateTls<TLS>(tls_key_, nullptr);
  }

  /** Get or create TID */
  HSHM_INLINE_CROSS_FUN
  hshm::ThreadId GetOrCreateTid(const hipc::MemContext &ctx) {
    hshm::ThreadId tid = ctx.tid_;
    if (tid.IsNull()) {
      TLS *tls = HERMES_THREAD_MODEL->GetTls<TLS>(tls_key_);
      if (!tls) {
        tid = header_->CreateTid();
        tls = header_->GetTls(tid);
        tls->alloc_ = this;
        tls->tid_ = tid;
        HERMES_THREAD_MODEL->SetTls(tls_key_, tls);
      } else {
        tid = tls->tid_;
      }
    }
    return tid;
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
    ThreadId tid = GetOrCreateTid(ctx);
    PageAllocator &page_alloc = (*header_->tls_)[tid.tid_];
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
      HERMES_THROW_ERROR(OUT_OF_MEMORY, size, GetCurrentlyAllocatedSize());
    }

    // Mark as allocated
    header_->total_alloc_.fetch_add(page_id.round_);
    OffsetPointer p = Convert<MpPage, OffsetPointer>(page);
    page->page_size_ = page_id.round_;
    page->tid_ = tid;
    page->SetAllocated();
    // HILOG(kInfo, "TIME: {}ns", timer.GetNsec());
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
    HERMES_THROW_ERROR(NOT_IMPLEMENTED, "AlignedAllocateOffset");
  }

  /**
   * Reallocate \a p pointer to \a new_size new size.
   *
   * @return whether or not the pointer p was changed
   * */
  HSHM_CROSS_FUN
  OffsetPointer ReallocateOffsetNoNullCheck(const hipc::MemContext &ctx,
                                            OffsetPointer p, size_t new_size) {
    FullPtr<char, OffsetPointer> new_ptr =
        ((AllocT *)this)->AllocateLocalPtr<char, OffsetPointer>(ctx, new_size);
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
      HERMES_THROW_ERROR(DOUBLE_FREE);
    }
    hdr->UnsetAllocated();
    header_->total_alloc_.fetch_sub(hdr->page_size_);
    ThreadId tid = GetOrCreateTid(ctx);
    PageAllocator &page_alloc = (*header_->tls_)[tid.tid_];
    page_alloc.Free(hdr);
  }

  /**
   * Get the current amount of data allocated. Can be used for leak
   * checking.
   * */
  HSHM_CROSS_FUN
  size_t GetCurrentlyAllocatedSize() { return header_->total_alloc_.load(); }

  /**
   * Create a globally-unique thread ID
   * */
  HSHM_CROSS_FUN
  void CreateTls(MemContext &ctx) { ctx.tid_ = GetOrCreateTid(ctx); }

  /**
   * Free a thread-local memory storage
   * */
  HSHM_CROSS_FUN
  void FreeTls(const MemContext &ctx) {
    ThreadId tid = GetOrCreateTid(ctx);
    if (tid.IsNull()) {
      return;
    }
    header_->FreeTid(tid);
    HERMES_THREAD_MODEL->SetTls<TLS>(tls_key_, nullptr);
  }
};

typedef BaseAllocator<_ScalablePageAllocator> ScalablePageAllocator;

}  // namespace hshm::ipc

#endif  // HERMES_MEMORY_ALLOCATOR_SCALABLE_PAGE_ALLOCATOR_H
