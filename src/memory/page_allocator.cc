/*
 * Copyright (C) 2022  SCS Lab <scslab@iit.edu>,
 * Luke Logan <llogan@hawk.iit.edu>,
 * Jaime Cernuda Garcia <jcernudagarcia@hawk.iit.edu>
 * Jay Lofstead <gflofst@sandia.gov>,
 * Anthony Kougkas <akougkas@iit.edu>,
 * Xian-He Sun <sun@iit.edu>
 *
 * This file is part of hermes
 *
 * hermes is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public
 * License along with this program.  If not, see
 * <http://www.gnu.org/licenses/>.
 */


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
