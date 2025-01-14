#ifndef HSHM_SHM_MEMORY_ALLOCATOR_TEST_ALLOCATOR_H
#define HSHM_SHM_MEMORY_ALLOCATOR_TEST_ALLOCATOR_H

#include <cmath>

#include "allocator.h"
#include "hermes_shm/data_structures/ipc/list.h"
#include "hermes_shm/data_structures/ipc/pair.h"
#include "hermes_shm/data_structures/ipc/ring_ptr_queue.h"
#include "hermes_shm/data_structures/ipc/ring_queue.h"
#include "hermes_shm/data_structures/ipc/vector.h"
#include "hermes_shm/memory/allocator/stack_allocator.h"
#include "hermes_shm/thread/lock.h"
#include "hermes_shm/util/logging.h"
#include "hermes_shm/util/timer.h"
#include "mp_page.h"
#include "page_allocator.h"

namespace hshm::ipc {

class _TestAllocator;
typedef BaseAllocator<_TestAllocator> TestAllocator;

struct _TestAllocatorHeader : public AllocatorHeader {
  typedef TlsAllocatorInfo<_TestAllocator> TLS;
  typedef hipc::PageAllocator<_TestAllocator, false, true> PageAllocator;
  typedef hipc::vector<PageAllocator, StackAllocator> PageAllocVec;
  typedef hipc::vector<hshm::size_t, StackAllocator> PageAllocIdVec;

  hipc::delay_ar<PageAllocVec> tls_;
  hipc::delay_ar<PageAllocIdVec> free_tids_;
  hipc::atomic<hshm::size_t> tid_heap_;
  hipc::atomic<hshm::size_t> total_alloc_;
  hipc::SpinLock lock_;

  HSHM_CROSS_FUN
  _TestAllocatorHeader() = default;

  HSHM_CROSS_FUN
  void Configure(AllocatorId alloc_id, size_t custom_header_size,
                 StackAllocator *alloc, size_t buffer_size,
                 size_t max_threads) {
    AllocatorHeader::Configure(alloc_id, AllocatorType::kTestAllocator,
                               custom_header_size);
    HSHM_MAKE_AR(tls_, alloc, max_threads, alloc);
    HSHM_MAKE_AR(free_tids_, alloc, max_threads);
    total_alloc_ = 0;
    tid_heap_ = 0;
  }

  HSHM_INLINE_CROSS_FUN
  hshm::ThreadId CreateTid() {
    ScopedSpinLock lock(lock_, 0);
    hshm::size_t tid = 0;
    if (free_tids_->size()) {
      tid = free_tids_->back();
      free_tids_->pop_back();
    } else {
      tid = tid_heap_.fetch_add(1);
    }
    HILOG(kInfo, "Allocating TID: {} (tid size: {})", tid, free_tids_->size());
    return hshm::ThreadId(tid);
  }

  HSHM_INLINE_CROSS_FUN
  void FreeTid(hshm::ThreadId tid) {
    ScopedSpinLock lock(lock_, 0);
    free_tids_->emplace_back(tid.tid_);
    HILOG(kInfo, "Freeing TID: {} (tid size: {})", tid, free_tids_->size());
  }

  HSHM_INLINE_CROSS_FUN
  TLS *GetTls(hshm::ThreadId tid) {
    return &(*tls_)[(size_t)tid.tid_].tls_info_;
  }
};

class _TestAllocator : public Allocator {
 public:
  HSHM_ALLOCATOR(_TestAllocator);

 private:
  typedef TlsAllocatorInfo<_TestAllocator> TLS;
  typedef _TestAllocatorHeader::PageAllocator PageAllocator;
  _TestAllocatorHeader *header_;
  StackAllocator alloc_;
  thread::ThreadLocalKey tls_key_;

 public:
  /**
   * Allocator constructor
   * */
  HSHM_CROSS_FUN
  _TestAllocator() : header_(nullptr) {}

  /**
   * Initialize the allocator in shared memory
   * */
  HSHM_CROSS_FUN
  void shm_init(AllocatorId id, size_t custom_header_size, char *buffer,
                size_t buffer_size, size_t max_threads = 1024) {
    type_ = AllocatorType::kTestAllocator;
    id_ = id;
    buffer_ = buffer;
    buffer_size_ = buffer_size;
    header_ = reinterpret_cast<_TestAllocatorHeader *>(buffer_);
    custom_header_ = reinterpret_cast<char *>(header_ + 1);
    size_t region_off = (custom_header_ - buffer_) + custom_header_size;
    size_t region_size = buffer_size_ - region_off;
    AllocatorId sub_id(id.bits_.major_, id.bits_.minor_ + 1);
    alloc_.shm_init(sub_id, 0, buffer + region_off, region_size);
    HSHM_MEMORY_MANAGER->RegisterSubAllocator(&alloc_);
    header_->Configure(id, custom_header_size, &alloc_, buffer_size,
                       max_threads);
    HSHM_THREAD_MODEL->CreateTls<TLS>(tls_key_, nullptr);
    alloc_.Align();
  }

  /**
   * Attach an existing allocator from shared memory
   * */
  HSHM_CROSS_FUN
  void shm_deserialize(char *buffer, size_t buffer_size) {
    buffer_ = buffer;
    buffer_size_ = buffer_size;
    header_ = reinterpret_cast<_TestAllocatorHeader *>(buffer_);
    type_ = header_->allocator_type_;
    id_ = header_->alloc_id_;
    custom_header_ = reinterpret_cast<char *>(header_ + 1);
    size_t region_off =
        (custom_header_ - buffer_) + header_->custom_header_size_;
    size_t region_size = buffer_size_ - region_off;
    alloc_.shm_deserialize(buffer + region_off, region_size);
    HSHM_MEMORY_MANAGER->RegisterSubAllocator(&alloc_);
    HSHM_THREAD_MODEL->CreateTls<TLS>(tls_key_, nullptr);
  }

  /** Get or create TID */
  HSHM_INLINE_CROSS_FUN
  hshm::ThreadId GetOrCreateTid(const hipc::MemContext &ctx) {
    hshm::ThreadId tid = ctx.tid_;
    if (tid.IsNull()) {
      TLS *tls = HSHM_THREAD_MODEL->GetTls<TLS>(tls_key_);
      if (!tls) {
        tid = header_->CreateTid();
        tls = header_->GetTls(tid);
        tls->alloc_ = this;
        tls->tid_ = tid;
        HSHM_THREAD_MODEL->SetTls(tls_key_, tls);
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
    PageAllocator &page_alloc = (*header_->tls_)[(size_t)tid.tid_];
    page = page_alloc.Allocate(page_id);

    // Case 2: Can we allocate of thread's heap?
    if (page == nullptr) {
      page = page_alloc.AllocateHeap(page_id);
      if (page) {
        page->tid_ = tid;
        page->page_size_ = page_id.round_;
      }
    }

    // // Case 3: Allocate from heap if no page found
    // if (page == nullptr) {
    //   OffsetPointer off = alloc_.SubAllocateOffset(page_id.round_);
    //   if (!off.IsNull()) {
    //     page = alloc_.Convert<MpPage>(off);
    //     page->tid_ = tid;
    //     page->page_size_ = page_id.round_;
    //   }
    // }

    // // Case 4: Completely out of memory
    if (page == nullptr) {
      HSHM_THROW_ERROR(OUT_OF_MEMORY, size, GetCurrentlyAllocatedSize());
    }

    // // Mark as allocated
    // header_->AddSize(page_id.round_);
    OffsetPointer p = Convert<MpPage, OffsetPointer>(page);
    // page->SetAllocated();
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
    FullPtr<char, OffsetPointer> new_ptr =
        GetAllocator()->AllocateLocalPtr<char, OffsetPointer>(ctx, new_size);
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
    // if (!hdr->IsAllocated()) {
    //   HSHM_THROW_ERROR(DOUBLE_FREE);
    // }
    // hdr->UnsetAllocated();
    // header_->SubSize(hdr->page_size_);
    PageAllocator &page_alloc = (*header_->tls_)[(size_t)hdr->tid_.tid_];
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
    HSHM_THREAD_MODEL->SetTls<TLS>(tls_key_, nullptr);
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_SHM_MEMORY_ALLOCATOR_TEST_ALLOCATOR_H