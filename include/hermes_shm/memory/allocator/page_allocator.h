#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_MEMORY_ALLOCATOR_PAGE_ALLOCATOR_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_MEMORY_ALLOCATOR_PAGE_ALLOCATOR_H_

#include <cmath>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/thread/lock/mutex.h"
#include "mp_page.h"
#include "stack_allocator.h"

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
  explicit PageId(size_t size) {
    orig_ = size;
#if HSHM_IS_HOST
    exp_ = (size_t)std::ceil(std::log2(size - sizeof(MpPage)));
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

template <typename AllocT, bool MPMC, bool LOCAL_HEAP>
class PageAllocator {
 public:
  typedef StackAllocator Alloc_;
  typedef TlsAllocatorInfo<AllocT> TLS;
  typedef hipc::lifo_list_queue<MpPage, Alloc_> LIFO_LIST;
  typedef hipc::mpsc_lifo_list_queue<MpPage, Alloc_> MPSC_LIFO_LIST;

 public:
  hipc::delay_ar<MPSC_LIFO_LIST> free_lists_[PageId::num_caches_];
  hipc::delay_ar<LIFO_LIST> fallback_list_;
  TLS tls_info_;
  HeapAllocator<MPMC> heap_;
  hipc::Mutex lock_;

 public:
  HSHM_INLINE_CROSS_FUN
  explicit PageAllocator(
      StackAllocator *alloc,
      size_t local_heap_size = hshm::Unit<size_t>::Kilobytes(1)) {
    for (size_t i = 0; i < PageId::num_caches_; ++i) {
      free_lists_[i].shm_init(alloc);
    }
    fallback_list_.shm_init(alloc);
    if constexpr (LOCAL_HEAP) {
      auto full_ptr = alloc->template Allocate<void, OffsetPointer>(HSHM_DEFAULT_MEM_CTX, local_heap_size);
      heap_.shm_init(full_ptr.shm_, local_heap_size);
    }
  }

  HSHM_INLINE_CROSS_FUN
  PageAllocator(const PageAllocator &other) {}

  HSHM_INLINE_CROSS_FUN
  PageAllocator(PageAllocator &&other) {}

  HSHM_INLINE_CROSS_FUN
  MpPage *AllocateHeap(const PageId &page_id) {
    if constexpr (LOCAL_HEAP) {
      if (page_id.exp_ < PageId::num_caches_) {
        OffsetPointer shm = heap_.AllocateOffset(page_id.round_);
        return tls_info_.alloc_->template Convert<MpPage>(shm);
      }
    }
    return nullptr;
  }

  HSHM_INLINE_CROSS_FUN
  MpPage *Allocate(const PageId &page_id) {
    if constexpr (!MPMC) {
      return AllocateMpsc(page_id);
    } else {
      hipc::ScopedMutex lock(lock_, 0);
      return AllocateMpsc(page_id);
    }
    // No page was cached
    return nullptr;
  }

  HSHM_INLINE_CROSS_FUN
  MpPage *AllocateMpsc(const PageId &page_id) {
    // Allocate cached page
    if (page_id.exp_ < PageId::num_caches_) {
      MPSC_LIFO_LIST &free_list = *free_lists_[page_id.exp_];
      MpPage *page = free_list.pop();
      return page;
    }
    // Allocate a large page size
    for (auto it = fallback_list_->begin(); it != fallback_list_->end(); ++it) {
      MpPage *page = *it;
      if (page->page_size_ >= page_id.round_) {
        fallback_list_->dequeue(it);
        return page;
      }
    }
    return nullptr;
  }

  HSHM_INLINE_CROSS_FUN
  void Free(OffsetPointer page_shm, MpPage *page) {
    PageId page_id(page->page_size_);
    if (page_id.exp_ < PageId::num_caches_) {
      free_lists_[page_id.exp_]->enqueue(page);
    } else {
      if constexpr (MPMC) {
        hipc::ScopedMutex lock(lock_, 0);
        fallback_list_->enqueue(page);
      } else {
        fallback_list_->enqueue(page);
      }
    }
  }
};

}  // namespace hshm::ipc

#endif