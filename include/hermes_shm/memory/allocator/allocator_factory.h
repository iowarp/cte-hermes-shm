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

#ifndef HSHM_MEMORY_ALLOCATOR_ALLOCATOR_FACTORY_H_
#define HSHM_MEMORY_ALLOCATOR_ALLOCATOR_FACTORY_H_

#include "allocator.h"
#include "allocator_factory_.h"
#include "gpu_stack_allocator.h"
#include "hermes_shm/memory/memory_manager_.h"
#include "malloc_allocator.h"
#include "scalable_page_allocator.h"
#include "stack_allocator.h"
#include "test_allocator.h"
#include "thread_local_allocator.h"

namespace hshm::ipc {

#define HSHM_ALLOC_DSRL_CASE(ALLOC_NAME)                                    \
  case AllocatorType::k##ALLOC_NAME: {                                      \
    auto full_ptr = HSHM_ROOT_ALLOC->NewObj<ALLOC_NAME>(HSHM_DEFAULT_MEM_CTX); \
    auto alloc = full_ptr.ptr_;                                             \
    alloc->shm_deserialize(backend);                                        \
    return alloc;                                                           \
  }

class AllocatorFactory {
 public:
  /**
   * Create a new memory allocator
   * */
  template <typename AllocT, typename... Args>
  HSHM_CROSS_FUN static AllocT* shm_init(AllocatorId alloc_id,
                                         size_t custom_header_size,
                                         MemoryBackend backend,
                                         Args&&... args) {
    auto full_ptr = HSHM_ROOT_ALLOC->NewObj<AllocT>(HSHM_DEFAULT_MEM_CTX);
    auto alloc = full_ptr.ptr_;
    alloc->backend_ = backend;
    alloc->shm_init(alloc_id, custom_header_size, backend,
                    std::forward<Args>(args)...);
    backend.SetHasAlloc();
    return alloc;
  }

  /**
   * Create a new memory allocator
   * */
  template <typename AllocT, typename... Args>
  HSHM_CROSS_FUN static AllocT* shm_init(AllocatorId alloc_id,
                                         size_t custom_header_size,
                                         MemoryBackend* backend,
                                         Args&&... args) {
    return shm_init<AllocT>(alloc_id, custom_header_size, *backend,
                            std::forward<Args>(args)...);
  }

  /**
   * Deserialize the allocator managing this backend.
   * */
  template <typename AllocT = Allocator>
  HSHM_CROSS_FUN static AllocT* shm_deserialize(const MemoryBackend& backend) {
    auto header_ = reinterpret_cast<AllocatorHeader*>(backend.md_);
    switch (header_->allocator_type_) {
      // Stack Allocator
      HSHM_ALLOC_DSRL_CASE(StackAllocator)
      HSHM_ALLOC_DSRL_CASE(GpuStackAllocator)
      HSHM_ALLOC_DSRL_CASE(MallocAllocator)
      HSHM_ALLOC_DSRL_CASE(ScalablePageAllocator)
      HSHM_ALLOC_DSRL_CASE(ThreadLocalAllocator)
      HSHM_ALLOC_DSRL_CASE(TestAllocator)
      default:
        return nullptr;
    }
  }

  /**
   * Deserialize the allocator managing this backend.
   * */
  template <typename AllocT = Allocator>
  HSHM_CROSS_FUN static AllocT* shm_deserialize(MemoryBackend* backend) {
    if (backend == nullptr) {
      return nullptr;
    }
    return shm_deserialize<AllocT>(*backend);
  }
};

}  // namespace hshm::ipc

#define HSHM_ALLOCATOR_FACTORY_DONE

#endif  // HSHM_MEMORY_ALLOCATOR_ALLOCATOR_FACTORY_H_
