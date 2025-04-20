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

#define HSHM_COMPILING_DLL
#define __HSHM_IS_COMPILING__

#include "hermes_shm/memory/memory_manager.h"

#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/memory/allocator/allocator_factory.h"
#include "hermes_shm/memory/backend/memory_backend_factory.h"
#include "hermes_shm/thread/thread_model_manager.h"
#include "hermes_shm/util/errors.h"
#include "hermes_shm/util/logging.h"

namespace hshm::ipc {

/** Create the root allocator */
HSHM_CROSS_FUN
MemoryManager::MemoryManager() { Init(); }

/** Initialize memory manager */
HSHM_CROSS_FUN
void MemoryManager::Init() {
  // System info
  HSHM_SYSTEM_INFO->RefreshInfo();

  // Initialize tables
  memset(backends_, 0, sizeof(backends_));
  memset(allocators_, 0, sizeof(allocators_));

  // Root backend
  ArrayBackend *root_backend = (ArrayBackend *)root_backend_space_;
  Allocator::ConstructObj(*root_backend);
  root_backend->shm_init(MemoryBackendId::GetRoot(), sizeof(root_alloc_data_),
                         root_alloc_data_);
  root_backend->Own();
  root_backend_ = root_backend;

  // Root allocator
  root_alloc_id_.bits_.major_ = 0;
  root_alloc_id_.bits_.minor_ = 0;
  StackAllocator *root_alloc = (StackAllocator *)root_alloc_space_;
  Allocator::ConstructObj(*root_alloc);
  root_alloc->shm_init(root_alloc_id_, 0, *root_backend_);
  root_alloc_ = root_alloc;
  default_allocator_ = root_alloc_;

  // Other allocators
  RegisterAllocatorNoScan(root_alloc_);
}

HSHM_DEFINE_GLOBAL_CROSS_PTR_VAR_CC(hshm::ipc::MemoryManager,
                                    hshmMemoryManager);

}  // namespace hshm::ipc

// TODO(llogan): Fix. A hack for HIP compiler to function
// I would love to spend more time figuring out why ROCm
// Fails to compile without this, but whatever.
#include "hermes_shm/introspect/system_info.cc"
