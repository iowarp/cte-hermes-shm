#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_MEMORY_ALLOCATOR_PAGE_ALLOCATOR_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_MEMORY_ALLOCATOR_PAGE_ALLOCATOR_H_

#include <cmath>

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

template <typename AllocT, bool THREAD_SAFE = false>
class PageAllocator {
 public:
  typedef StackAllocator Alloc_;
  typedef TlsAllocatorInfo<AllocT> TLS;
  typedef hipc::lifo_list_queue<MpPage, Alloc_> LIFO_LIST;
  typedef hipc::mpmc_lifo_list_queue<MpPage, Alloc_> MPMC_LIFO_LIST;
  typedef
      typename std::conditional<THREAD_SAFE, LIFO_LIST, MPMC_LIFO_LIST>::type
          LIST;
  hipc::Mutex lock_;

 public:
  hipc::delay_ar<LIST> free_lists_[PageId::num_caches_];
  hipc::delay_ar<LIFO_LIST> fallback_list_;
  TLS tls_info_;

 public:
  HSHM_INLINE_CROSS_FUN
  explicit PageAllocator(StackAllocator *alloc) {
    for (size_t i = 0; i < PageId::num_caches_; ++i) {
      HSHM_MAKE_AR0(free_lists_[i], alloc);
    }
    HSHM_MAKE_AR0(fallback_list_, alloc);
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
    if constexpr (THREAD_SAFE) {
      hipc::ScopedMutex lock(lock_, 0);
      for (auto it = fallback_list_->begin(); it != fallback_list_->end();
           ++it) {
        MpPage *page = *it;
        if (page->page_size_ >= page_id.round_) {
          fallback_list_->dequeue(it);
          return page;
        }
      }
    } else {
      for (auto it = fallback_list_->begin(); it != fallback_list_->end();
           ++it) {
        MpPage *page = *it;
        if (page->page_size_ >= page_id.round_) {
          fallback_list_->dequeue(it);
          return page;
        }
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
      fallback_list_->enqueue(page);
    }
  }
};

}  // namespace hshm::ipc

#endif