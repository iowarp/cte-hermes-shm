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


#include <hermes_shm/memory/allocator/scalable_page_allocator.h>
#include <hermes_shm/memory/allocator/mp_page.h>

namespace hermes::ipc {

void ScalablePageAllocator::shm_init(allocator_id_t id,
                                     size_t custom_header_size,
                                     char *buffer,
                                     size_t buffer_size,
                                     RealNumber coalesce_trigger,
                                     size_t coalesce_window) {
  buffer_ = buffer;
  buffer_size_ = buffer_size;
  header_ = reinterpret_cast<ScalablePageAllocatorHeader*>(buffer_);
  custom_header_ = reinterpret_cast<char*>(header_ + 1);
  size_t region_off = (custom_header_ - buffer_) + custom_header_size;
  size_t region_size = buffer_size_ - region_off;
  alloc_.shm_init(id, 0, buffer + region_off, region_size);
  header_->Configure(id, custom_header_size, &alloc_,
                     buffer_size, coalesce_trigger, coalesce_window);
  free_lists_->shm_deserialize(header_->free_lists_.internal_ref(&alloc_));
  // Cache every power-of-two between 16B and 16KB
  size_t ncpu = HERMES_SYSTEM_INFO->ncpu_;
  free_lists_->resize(ncpu * num_free_lists_);
  for (size_t i = 0; i < HERMES_SYSTEM_INFO->ncpu_; ++i) {
    for (size_t j = 0; j < num_caches_; ++j) {
      hipc::ShmRef<pair<FreeListStats, iqueue<MpPage>>>
        free_list = (*free_lists_)[i * ncpu + j];
      free_list->first_->page_size_ = 16 * (1 << j);
      free_list->first_->count_ = 0;
    }
  }
}

void ScalablePageAllocator::shm_deserialize(char *buffer,
                                            size_t buffer_size) {
  buffer_ = buffer;
  buffer_size_ = buffer_size;
  header_ = reinterpret_cast<ScalablePageAllocatorHeader*>(buffer_);
  custom_header_ = reinterpret_cast<char*>(header_ + 1);
  size_t region_off = (custom_header_ - buffer_) + header_->custom_header_size_;
  size_t region_size = buffer_size_ - region_off;
  alloc_.shm_deserialize(buffer + region_off, region_size);
  free_lists_->shm_deserialize(header_->free_lists_.internal_ref(&alloc_));
}

size_t ScalablePageAllocator::GetCurrentlyAllocatedSize() {
  return header_->total_alloc_;
}

/** Round a number up to the nearest page size. */
size_t ScalablePageAllocator::RoundUp(size_t num, int &exp) {
  int round;
  for (exp = 0; exp < num_caches_; ++exp) {
    round = 1 << (exp + 4);
    if (num < round) {
      return round;
    }
  }
  return num;
}

OffsetPointer ScalablePageAllocator::AllocateOffset(size_t size) {
  MpPage *page = nullptr;
  size_t size_mp = size + sizeof(MpPage);

  // Case 1: There is a page (of nearly this size) cached
  if (size_mp <= KILOBYTES(16)) {
    int exp;
    size_mp = RoundUp(size_mp, exp);
    hipc::ShmRef<pair<FreeListStats, iqueue<MpPage>>> free_list =
      (*free_lists_)[exp];
    if (free_list->second_->size()) {
      page = free_list->second_->dequeue();
    }
  } else {
    hipc::ShmRef<pair<FreeListStats, iqueue<MpPage>>> free_list =
      (*free_lists_)[num_caches_];
    for (size_t i = 0; i < free_list->second_->size(); ++i) {

    }
    for (MpPage *fit_page : (*free_list->second_)) {
      if (fit_page->page_size_ >= size) {
        size_t rem = fit_page->page_size_ - size;
        if (rem >= KILOBYTES(16)) {

        }
      }
    }
  }

  // Case 2: Coalesce if enough space is being wasted
  if (page == nullptr) {

  }

  // Case 3: Allocate from stack if no page found
  if (page == nullptr){
    page = alloc_.Convert<MpPage>(alloc_.AllocateOffset(size) - sizeof(MpPage));
  }

  // Case 4: Completely out of memory
  if (page == nullptr) {
    throw OUT_OF_MEMORY;
  }

  // Mark as allocated
  header_->total_alloc_.fetch_add(page->page_size_);
  auto p = Convert<MpPage, OffsetPointer>(page);
  page->SetAllocated();
  return p + sizeof(MpPage);
}

OffsetPointer ScalablePageAllocator::AlignedAllocateOffset(size_t size,
                                                        size_t alignment) {
  throw ALIGNED_ALLOC_NOT_SUPPORTED.format();
}

OffsetPointer ScalablePageAllocator::ReallocateOffsetNoNullCheck(OffsetPointer p,
                                                          size_t new_size) {
  throw ALIGNED_ALLOC_NOT_SUPPORTED.format();
}

void ScalablePageAllocator::FreeOffsetNoNullCheck(OffsetPointer p) {
  // Mark as free
  auto hdr_offset = p - sizeof(MpPage);
  auto hdr = Convert<MpPage>(hdr_offset);
  if (!hdr->IsAllocated()) {
    throw DOUBLE_FREE.format();
  }
  hdr->UnsetAllocated();
  header_->total_alloc_.fetch_sub(hdr->page_size_);

  // Append to a free list
  for (hipc::ShmRef<iqueue<MpPage>> free_list : *free_lists_) {
    if (free_list->size()) {
      MpPage *page = free_list->peek();
      if (page->page_size_ != hdr->page_size_) {
        continue;
      }
    }
    free_list->enqueue(hdr);
    return;
  }
}

}  // namespace hermes::ipc
