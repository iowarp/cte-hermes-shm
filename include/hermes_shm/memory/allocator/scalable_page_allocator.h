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

struct FreeListSetIpc : public ShmContainer {
  SHM_CONTAINER_TEMPLATE(FreeListSetIpc, FreeListSetIpc)
  ShmArchive<vector<pair<Mutex, iqueue<MpPage>>>> lists_;
  std::atomic<uint16_t> rr_free_;
  std::atomic<uint16_t> rr_alloc_;

  /** SHM constructor. Default. */
  HSHM_CROSS_FUN
  explicit FreeListSetIpc(Allocator *alloc) {
    shm_init_container(alloc);
    HSHM_MAKE_AR0(lists_, alloc)
    SetNull();
  }

  /** SHM emplace constructor */
  HSHM_CROSS_FUN
  explicit FreeListSetIpc(Allocator *alloc, size_t conc) {
    shm_init_container(alloc);
    HSHM_MAKE_AR(lists_, alloc, conc)
    SetNull();
  }

  /** SHM copy constructor. */
  HSHM_CROSS_FUN
  explicit FreeListSetIpc(Allocator *alloc, const FreeListSetIpc &other) {
    shm_init_container(alloc);
    SetNull();
  }

  /** SHM copy assignment operator */
  HSHM_CROSS_FUN
  FreeListSetIpc& operator=(const FreeListSetIpc &other) {
    if (this != &other) {
      shm_destroy();
      SetNull();
    }
    return *this;
  }

  /** Destructor. */
  HSHM_CROSS_FUN
  void shm_destroy_main() {
    lists_->shm_destroy();
  }

  /** Check if Null */
  HSHM_INLINE_CROSS_FUN bool IsNull() {
    return false;
  }

  /** Set to null */
  HSHM_INLINE_CROSS_FUN void SetNull() {
    rr_free_ = 0;
    rr_alloc_ = 0;
  }
};

struct FreeListSet {
  std::vector<std::pair<Mutex*, iqueue<MpPage>*>> lists_;
  std::atomic<uint16_t> *rr_free_;
  std::atomic<uint16_t> *rr_alloc_;
};

struct ScalablePageAllocatorHeader : public AllocatorHeader {
  ShmArchive<vector<FreeListSetIpc>> free_lists_;
  std::atomic<size_t> total_alloc_;
  size_t coalesce_trigger_;
  size_t coalesce_window_;

  HSHM_CROSS_FUN
  ScalablePageAllocatorHeader() = default;

  HSHM_CROSS_FUN
  void Configure(allocator_id_t alloc_id,
                 size_t custom_header_size,
                 Allocator *alloc,
                 size_t buffer_size,
                 RealNumber coalesce_trigger,
                 size_t coalesce_window) {
    AllocatorHeader::Configure(alloc_id,
                               AllocatorType::kScalablePageAllocator,
                               custom_header_size);
    HSHM_MAKE_AR0(free_lists_, alloc)
    total_alloc_ = 0;
    coalesce_trigger_ = (coalesce_trigger * buffer_size).as_int();
    coalesce_window_ = coalesce_window;
  }
};

class ScalablePageAllocator : public Allocator {
 private:
  ScalablePageAllocatorHeader *header_;
  std::vector<FreeListSet> free_lists_;
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
   * Get the ID of this allocator from shared memory
   * */
  HSHM_CROSS_FUN
  allocator_id_t &GetId() override {
    return header_->allocator_id_;
  }

  /**
   * Initialize the allocator in shared memory
   * */
  HSHM_CROSS_FUN
  void shm_init(allocator_id_t id,
                size_t custom_header_size,
                char *buffer,
                size_t buffer_size,
                RealNumber coalesce_trigger = RealNumber(1, 5),
                size_t coalesce_window = MEGABYTES(1)) {
    buffer_ = buffer;
    buffer_size_ = buffer_size;
    header_ = reinterpret_cast<ScalablePageAllocatorHeader*>(buffer_);
    custom_header_ = reinterpret_cast<char*>(header_ + 1);
    size_t region_off = (custom_header_ - buffer_) + custom_header_size;
    size_t region_size = buffer_size_ - region_off;
    allocator_id_t sub_id(id.bits_.major_, id.bits_.minor_ + 1);
    alloc_.shm_init(sub_id, 0, buffer + region_off, region_size);
    HERMES_MEMORY_MANAGER->RegisterAllocator(&alloc_);
    header_->Configure(id, custom_header_size, &alloc_,
                       buffer_size, coalesce_trigger, coalesce_window);
    vector<FreeListSetIpc> *free_lists = header_->free_lists_.get();
    size_t ncpu = HERMES_SYSTEM_INFO->ncpu_;
    free_lists->resize(num_free_lists_, ncpu);
    CacheFreeLists();
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
    custom_header_ = reinterpret_cast<char*>(header_ + 1);
    size_t region_off = (custom_header_ - buffer_) + header_->custom_header_size_;
    size_t region_size = buffer_size_ - region_off;
    alloc_.shm_deserialize(buffer + region_off, region_size);
    HERMES_MEMORY_MANAGER->RegisterAllocator(&alloc_);
    CacheFreeLists();
  }

  /**
   * Cache the free lists
   * */
  HSHM_INLINE_CROSS_FUN void CacheFreeLists() {
    vector<FreeListSetIpc> *free_lists = header_->free_lists_.get();
    free_lists_.reserve(free_lists->size());
    // Iterate over page cache sets
    for (FreeListSetIpc &free_list_set_ipc : *free_lists) {
      free_lists_.emplace_back();
      FreeListSet &free_list_set = free_lists_.back();
      vector<pair<Mutex, iqueue<MpPage>>> &lists_ipc =
        *free_list_set_ipc.lists_;
      std::vector<std::pair<Mutex*, iqueue<MpPage>*>> &lists =
        free_list_set.lists_;
      free_list_set.lists_.reserve(free_list_set_ipc.lists_->size());
      free_list_set.rr_alloc_ = &free_list_set_ipc.rr_alloc_;
      free_list_set.rr_free_ = &free_list_set_ipc.rr_free_;
      // Iterate over single page cache lane
      for (pair<Mutex, iqueue<MpPage>> &free_list_pair_ipc : lists_ipc) {
        Mutex &lock_ipc = free_list_pair_ipc.GetFirst();
        iqueue<MpPage> &free_list_ipc = free_list_pair_ipc.GetSecond();
        lists.emplace_back(&lock_ipc, &free_list_ipc);
      }
    }
  }

  /**
   * Allocate a memory of \a size size. The page allocator cannot allocate
   * memory larger than the page size.
   * */
  HSHM_CROSS_FUN
  OffsetPointer AllocateOffset(size_t size) override {
    MpPage *page = nullptr;
    size_t exp;
    size_t size_mp = RoundUp(size + sizeof(MpPage), exp);

    // Case 1: Can we re-use an existing page?
    page = CheckLocalCaches(size_mp, exp);

    // Case 2: Coalesce if enough space is being wasted
    // if (page == nullptr) {}

    // Case 3: Allocate from stack if no page found
    if (page == nullptr) {
      auto off = alloc_.AllocateOffset(size_mp);
      if (!off.IsNull()) {
        page = alloc_.Convert<MpPage>(off - sizeof(MpPage));
      }
    }

    // Case 4: Completely out of memory
    if (page == nullptr) {
      throw OUT_OF_MEMORY;
    }

    // Mark as allocated
    page->page_size_ = size_mp;
    header_->total_alloc_.fetch_add(page->page_size_);
    auto p = Convert<MpPage, OffsetPointer>(page);
    page->SetAllocated();
    return p + sizeof(MpPage);
  }

 private:
  /** Check if a cached page on this core can be re-used */
  HSHM_INLINE_CROSS_FUN MpPage* CheckLocalCaches(size_t size_mp, size_t exp) {
    MpPage *page;

    // Check the small buffer caches
    if (size_mp <= max_cached_size_) {
      // Get buffer cache at exp
      FreeListSet &free_list_set = free_lists_[exp];
      uint16_t conc = free_list_set.rr_alloc_->fetch_add(1) %
        free_list_set.lists_.size();
      std::pair<Mutex*, iqueue<MpPage>*> free_list_pair =
        free_list_set.lists_[conc];
      Mutex &lock = *free_list_pair.first;
      iqueue<MpPage> &free_list = *free_list_pair.second;
      ScopedMutex scoped_lock(lock, 0);

      // Check buffer cache
      if (free_list.size()) {
        page = free_list.dequeue();
        return page;
      } else {
        return nullptr;
      }
    } else {
      // Get buffer cache at exp
      FreeListSet &free_list_set = free_lists_[num_caches_];
      uint16_t conc = free_list_set.rr_alloc_->fetch_add(1) %
        free_list_set.lists_.size();
      std::pair<Mutex*, iqueue<MpPage>*> free_list_pair =
        free_list_set.lists_[conc];
      Mutex &lock = *free_list_pair.first;
      iqueue<MpPage> &free_list = *free_list_pair.second;
      ScopedMutex scoped_lock(lock, 0);

      // Check the arbitrary buffer cache
      page = FindFirstFit(size_mp,
                          free_list);
      return page;
    }
  }

  /** Find the first fit of an element in a free list */
  HSHM_INLINE_CROSS_FUN MpPage* FindFirstFit(size_t size_mp,
                                          iqueue<MpPage> &free_list) {
    for (auto iter = free_list.begin(); iter != free_list.end(); ++iter) {
      MpPage *fit_page = *iter;
      MpPage *rem_page;
      if (fit_page->page_size_ >= size_mp) {
        DividePage(free_list, fit_page, rem_page, size_mp, 0);
        free_list.dequeue(iter);
        if (rem_page) {
          free_list.enqueue(rem_page);
        }
        return fit_page;
      }
    }
    return nullptr;
  }

  /**
   * Divide a page into smaller pages and cache them
   * */
  HSHM_CROSS_FUN
  void DividePage(iqueue<MpPage> &free_list,
                  MpPage *fit_page,
                  MpPage *&rem_page,
                  size_t size_mp,
                  size_t max_divide) {
    // Get space remaining after size_mp is allocated
    size_t rem_size;
    rem_size = fit_page->page_size_ - size_mp;

    // Case 1: The remaining size can't be cached
    rem_page = nullptr;
    if (rem_size < min_cached_size_) {
      return;
    }

    // Case 2: Divide the remaining space into units of size_mp
    fit_page->page_size_ = size_mp;
    rem_page = (MpPage *) ((char *) fit_page + size_mp);
    if (max_divide > 0 && rem_size >= size_mp) {
      size_t num_divisions = (rem_size - size_mp) / size_mp;
      if (num_divisions > max_divide) { num_divisions = max_divide; }
      for (size_t i = 0; i < num_divisions; ++i) {
        rem_page->page_size_ = size_mp;
        rem_page->flags_.Clear();
        rem_page->off_ = 0;
        free_list.enqueue(rem_page);
        rem_page = (MpPage *) ((char *) rem_page + size_mp);
        rem_size -= size_mp;
      }
    }

    // Case 3: There is still remaining space after the divisions
    if (rem_size > 0) {
      rem_page->page_size_ = rem_size;
      rem_page->flags_.Clear();
      rem_page->off_ = 0;
    } else {
      rem_page = nullptr;
    }
  }


 public:
  /**
   * Allocate a memory of \a size size, which is aligned to \a
   * alignment.
   * */
  HSHM_CROSS_FUN
  OffsetPointer AlignedAllocateOffset(size_t size, size_t alignment) override {
    throw ALIGNED_ALLOC_NOT_SUPPORTED.format();
  }

  /**
   * Reallocate \a p pointer to \a new_size new size.
   *
   * @return whether or not the pointer p was changed
   * */
  HSHM_CROSS_FUN
  OffsetPointer ReallocateOffsetNoNullCheck(
    OffsetPointer p, size_t new_size) override {
    OffsetPointer new_p;
    void *ptr = AllocatePtr<void*, OffsetPointer>(new_size, new_p);
    MpPage *hdr = Convert<MpPage>(p - sizeof(MpPage));
    void *old = (void*)(hdr + 1);
    memcpy(ptr, old, hdr->page_size_ - sizeof(MpPage));
    FreeOffsetNoNullCheck(p);
    return new_p;
  }

  /**
   * Free \a ptr pointer. Null check is performed elsewhere.
   * */
  HSHM_CROSS_FUN
  void FreeOffsetNoNullCheck(OffsetPointer p) override {
    // Mark as free
    auto hdr_offset = p - sizeof(MpPage);
    auto hdr = Convert<MpPage>(hdr_offset);
    if (!hdr->IsAllocated()) {
      throw DOUBLE_FREE.format();
    }
    hdr->UnsetAllocated();
    header_->total_alloc_.fetch_sub(hdr->page_size_);
    size_t exp;
    RoundUp(hdr->page_size_, exp);

    // Append to small buffer cache free list
    if (hdr->page_size_ <= max_cached_size_) {
      // Get buffer cache at exp
      FreeListSet &free_list_set = free_lists_[exp];
      uint16_t conc = free_list_set.rr_free_->fetch_add(1) %
          free_list_set.lists_.size();
      std::pair<Mutex*, iqueue<MpPage>*> free_list_pair =
          free_list_set.lists_[conc];
      Mutex &lock = *free_list_pair.first;
      iqueue<MpPage> &free_list = *free_list_pair.second;
      ScopedMutex scoped_lock(lock, 0);
      free_list.enqueue(hdr);
    } else {
      // Get buffer cache at exp
      FreeListSet &free_list_set = free_lists_[num_caches_];
      uint16_t conc = free_list_set.rr_free_->fetch_add(1) %
          free_list_set.lists_.size();
      std::pair<Mutex*, iqueue<MpPage>*> free_list_pair =
          free_list_set.lists_[conc];
      Mutex &lock = *free_list_pair.first;
      iqueue<MpPage> &free_list = *free_list_pair.second;
      ScopedMutex scoped_lock(lock, 0);
      free_list.enqueue(hdr);
    }
  }

  /**
   * Get the current amount of data allocated. Can be used for leak
   * checking.
   * */
  HSHM_CROSS_FUN
  size_t GetCurrentlyAllocatedSize() override {
    return header_->total_alloc_;
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
