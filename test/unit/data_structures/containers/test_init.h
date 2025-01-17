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

#ifndef HSHM_TEST_UNIT_DATA_STRUCTURES_TEST_INIT_H_
#define HSHM_TEST_UNIT_DATA_STRUCTURES_TEST_INIT_H_

#include "hermes_shm/data_structures/all.h"

using hshm::ipc::Allocator;
using hshm::ipc::AllocatorId;
using hshm::ipc::AllocatorType;
using hshm::ipc::MemoryBackend;
using hshm::ipc::MemoryBackendType;
using hshm::ipc::Pointer;
using hshm::ipc::PosixShmMmap;

using hshm::ipc::Allocator;
using hshm::ipc::AllocatorId;
using hshm::ipc::AllocatorType;
using hshm::ipc::MemoryBackend;
using hshm::ipc::MemoryBackendType;
using hshm::ipc::MemoryManager;
using hshm::ipc::Pointer;

GLOBAL_CONST AllocatorId MAIN_alloc_id(1, 0);

template <typename AllocT>
void Pretest() {
  std::string shm_url = "test_allocators";
  auto mem_mngr = HSHM_MEMORY_MANAGER;
  mem_mngr->UnregisterAllocator(MAIN_ALLOC_ID);
  mem_mngr->DestroyBackend(hipc::MemoryBackendId::GetRoot());
  mem_mngr->CreateBackend<PosixShmMmap>(hipc::MemoryBackendId::Get(0),
                                        hshm::Unit<size_t>::Megabytes(100),
                                        shm_url);
  mem_mngr->CreateAllocator<AllocT>(hipc::MemoryBackendId::Get(0),
                                    MAIN_ALLOC_ID, 0);
}

void Posttest();

#endif  // HSHM_TEST_UNIT_DATA_STRUCTURES_TEST_INIT_H_
