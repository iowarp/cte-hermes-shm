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


#include <hermes_shm/memory/allocator/page_allocator.h>
// #include <hermes_shm/data_structures/thread_unsafe/list.h>

namespace hermes::ipc {

void PageAllocator::shm_init(MemoryBackend *backend,
                             allocator_id_t id,
                             size_t custom_header_size,
                             size_t page_size) {
  backend_ = backend;
  ibackend_.shm_init(backend_->data_size_, backend_->data_);
  ialloc_.shm_init(&ibackend_, allocator_id_t(0,1),
                   sizeof(PageAllocatorHeader));
  header_ = ialloc_.GetCustomHeader<PageAllocatorHeader>();
  header_->Configure(id, custom_header_size, page_size);
  custom_header_ = ialloc_.AllocatePtr<char>(custom_header_size,
                                             header_->custom_header_ptr_);
}

void PageAllocator::shm_deserialize(MemoryBackend *backend) {
  backend_ = backend;
  ibackend_.shm_deserialize(backend_->data_);
  ialloc_.shm_deserialize(&ibackend_);
  header_ = ialloc_.GetCustomHeader<PageAllocatorHeader>();
  custom_header_ = ialloc_.Convert<char>(header_->custom_header_ptr_);
}

OffsetPointer PageAllocator::AllocateOffset(size_t size) {
  if (size > header_->page_size_) {
    throw PAGE_SIZE_UNSUPPORTED.format(size);
  }

  // Try re-using cached page
  OffsetPointer p;

  // Try allocating off segment
  if (p.IsNull()) {
    p = ialloc_.AllocateOffset(size);
  }

  // Return
  if (!p.IsNull()) {
    header_->total_alloced_ += header_->page_size_;
    return p;
  }
  return OffsetPointer::GetNull();
}

OffsetPointer PageAllocator::AlignedAllocateOffset(size_t size,
                                                   size_t alignment) {
  throw ALIGNED_ALLOC_NOT_SUPPORTED.format();
}

OffsetPointer PageAllocator::ReallocateOffsetNoNullCheck(OffsetPointer p,
                                                         size_t new_size) {
  if (new_size > header_->page_size_) {
    throw PAGE_SIZE_UNSUPPORTED.format(new_size);
  }
  return p;
}

void PageAllocator::FreeOffsetNoNullCheck(OffsetPointer p) {
  header_->total_alloced_ -= header_->page_size_;
}

size_t PageAllocator::GetCurrentlyAllocatedSize() {
  return 0;
}

}  // namespace hermes::ipc
