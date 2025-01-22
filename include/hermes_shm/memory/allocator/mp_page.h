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

#ifndef HSHM_INCLUDE_HSHM_MEMORY_ALLOCATOR_MP_PAGE_H_
#define HSHM_INCLUDE_HSHM_MEMORY_ALLOCATOR_MP_PAGE_H_

#include "hermes_shm/data_structures/ipc/lifo_list_queue.h"
#include "hermes_shm/data_structures/ipc/mpsc_lifo_list_queue.h"

namespace hshm::ipc {

struct MpPage : public atomic_list_queue_entry {
  ibitfield flags_; /**< Flags of the page (e.g., free/alloc) */
  /** Offset from the start of the page to the beginning of this header */
  u32 off_;
  ThreadId tid_;     /**< The thread ID that allocated the page */
  size_t page_size_; /**< The total size of the page allocated */

  HSHM_INLINE_CROSS_FUN void SetAllocated() { flags_.SetBits(0x1); }

  HSHM_INLINE_CROSS_FUN void UnsetAllocated() { flags_.Clear(); }

  HSHM_INLINE_CROSS_FUN bool IsAllocated() const { return flags_.All(0x1); }
};

}  // namespace hshm::ipc

#endif  // HSHM_INCLUDE_HSHM_MEMORY_ALLOCATOR_MP_PAGE_H_
