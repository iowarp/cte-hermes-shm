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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_MEMORY_ALLOCATOR_HEAP_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_MEMORY_ALLOCATOR_HEAP_H_

#include "allocator.h"
#include "hermes_shm/thread/lock.h"

namespace hshm::ipc {

template <bool ATOMIC>
struct HeapAllocator {
  hshm::size_t region_off_;
  hipc::opt_atomic<hshm::size_t, ATOMIC> heap_off_;
  hshm::size_t heap_size_;

  /** Default constructor */
  HSHM_CROSS_FUN
  HeapAllocator() : region_off_(0), heap_off_(0), heap_size_(0) {}

  /** Emplace constructor */
  HSHM_CROSS_FUN
  explicit HeapAllocator(size_t region_off, size_t heap_size)
      : region_off_(region_off), heap_off_(0), heap_size_(heap_size) {}

  /** Explicit initialization */
  HSHM_CROSS_FUN
  void shm_init(size_t region_off, size_t heap_size) {
    region_off_ = region_off;
    heap_off_ = 0;
    heap_size_ = heap_size;
  }

  /** Explicit initialization */
  HSHM_CROSS_FUN
  void shm_init(const OffsetPointer &region_off, size_t heap_size) {
    region_off_ = region_off.off_.load();
    heap_off_ = 0;
    heap_size_ = heap_size;
  }

  /** Allocate off heap */
  HSHM_INLINE_CROSS_FUN OffsetPointer AllocateOffset(size_t size) {
    // if (size % 64 != 0) {
    //   size = (size + 63) & ~63;
    // }
    hshm::size_t off = heap_off_.fetch_add((hshm::size_t)size);
    if (off + size > heap_size_) {
      // HSHM_THROW_ERROR(OUT_OF_MEMORY, size, heap_size_);
      return OffsetPointer::GetNull();
    }
    return OffsetPointer((size_t)(region_off_ + off));
  }

  /** Copy assignment operator */
  HSHM_CROSS_FUN
  HeapAllocator &operator=(const HeapAllocator &other) {
    region_off_ = other.region_off_;
    heap_off_ = other.heap_off_.load();
    heap_size_ = other.heap_size_;
    return *this;
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_MEMORY_ALLOCATOR_HEAP_H_
