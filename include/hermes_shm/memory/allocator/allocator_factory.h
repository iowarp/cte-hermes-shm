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


#ifndef HERMES_MEMORY_ALLOCATOR_ALLOCATOR_FACTORY_H_
#define HERMES_MEMORY_ALLOCATOR_ALLOCATOR_FACTORY_H_

#include "allocator.h"
#include "stack_allocator.h"
#include "malloc_allocator.h"
#include "scalable_page_allocator.h"
#include "hermes_shm/memory/memory_manager_.h"

namespace hshm::ipc {

class AllocatorFactory {
 public:
  /**
   * Create a new memory allocator
   * */
  template<typename AllocT, typename ...Args>
  static Allocator* shm_init(AllocatorId alloc_id,
                             size_t custom_header_size,
                             MemoryBackend *backend,
                             Args&& ...args) {
    if constexpr(std::is_same_v<StackAllocator, AllocT>) {
      // StackAllocator
      auto alloc = HERMES_MEMORY_MANAGER->GetRootAllocator()->NewObj<StackAllocator>();
      alloc->shm_init(alloc_id,
                      custom_header_size,
                      backend->data_,
                      backend->data_size_,
                      std::forward<Args>(args)...);
      return alloc;
    } else if constexpr(std::is_same_v<MallocAllocator, AllocT>) {
      // Malloc Allocator
      auto alloc = HERMES_MEMORY_MANAGER->GetRootAllocator()->NewObj<MallocAllocator>();
      alloc->shm_init(alloc_id,
                      custom_header_size,
                      backend->data_size_,
                      std::forward<Args>(args)...);
      return alloc;
    } else if constexpr(std::is_same_v<ScalablePageAllocator, AllocT>) {
      // Scalable Page Allocator
      auto alloc = HERMES_MEMORY_MANAGER->GetRootAllocator()->NewObj<ScalablePageAllocator>();
      alloc->shm_init(alloc_id,
                      custom_header_size,
                      backend->data_,
                      backend->data_size_,
                      std::forward<Args>(args)...);
      return alloc;
    } else {
      // Default
      static_assert("Not a valid allocator");
    }
  }

  /**
   * Deserialize the allocator managing this backend.
   * */
  static Allocator* shm_deserialize(MemoryBackend *backend) {
    if (backend == nullptr) {
      return nullptr;
    }
    auto header_ = reinterpret_cast<AllocatorHeader*>(backend->data_);
    switch (header_->allocator_type_) {
      // Stack Allocator
      case AllocatorType::kStackAllocator: {
        auto alloc = HERMES_MEMORY_MANAGER->GetRootAllocator()->NewObj<StackAllocator>();
        alloc->shm_deserialize(backend->data_,
                               backend->data_size_);
        return alloc;
      }
      // Malloc Allocator
      case AllocatorType::kMallocAllocator: {
        auto alloc = HERMES_MEMORY_MANAGER->GetRootAllocator()->NewObj<MallocAllocator>();
        alloc->shm_deserialize(backend->data_,
                               backend->data_size_);
        return alloc;
      }
      // Scalable Page Allocator
      case AllocatorType::kScalablePageAllocator: {
        auto alloc = HERMES_MEMORY_MANAGER->GetRootAllocator()->NewObj<ScalablePageAllocator>();
        alloc->shm_deserialize(backend->data_,
                               backend->data_size_);
        return alloc;
      }
      default: return nullptr;
    }
  }

  /**
   * Attach the allocator
   * */
  HSHM_CROSS_FUN
  static Allocator* shm_attach(Allocator *other) {
    Allocator *alloc = nullptr;
    switch (other->type_) {
      // Stack Allocator
      case AllocatorType::kStackAllocator: {
        alloc = hipc::make_mptr<StackAllocator>(
            HERMES_MEMORY_MANAGER->GetRootAllocator(),
            *(StackAllocator*)other).get();
        break;
      }

      // Scalable Page Allocator
      case AllocatorType::kScalablePageAllocator: {
        alloc = hipc::make_mptr<ScalablePageAllocator>(
            HERMES_MEMORY_MANAGER->GetRootAllocator(),
            *(ScalablePageAllocator*)other).get();
        break;
      }

      // Malloc Allocator
      case AllocatorType::kMallocAllocator: {
        alloc = hipc::make_mptr<MallocAllocator>(
            HERMES_MEMORY_MANAGER->GetRootAllocator(),
            *(MallocAllocator*)other).get();
        break;
      }
    }
    return alloc;
  }
};

}  // namespace hshm::ipc

#endif  // HERMES_MEMORY_ALLOCATOR_ALLOCATOR_FACTORY_H_
