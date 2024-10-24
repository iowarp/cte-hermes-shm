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


#ifndef HERMES_TEST_UNIT_DATA_STRUCTURES_TEST_INIT_H_
#define HERMES_TEST_UNIT_DATA_STRUCTURES_TEST_INIT_H_

#include "hermes_shm/data_structures/data_structure.h"

using hshm::ipc::PosixShmMmap;
using hshm::ipc::MemoryBackendType;
using hshm::ipc::MemoryBackend;
using hshm::ipc::allocator_id_t;
using hshm::ipc::AllocatorType;
using hshm::ipc::Allocator;
using hshm::ipc::Pointer;

using hshm::ipc::MemoryBackendType;
using hshm::ipc::MemoryBackend;
using hshm::ipc::allocator_id_t;
using hshm::ipc::AllocatorType;
using hshm::ipc::Allocator;
using hshm::ipc::MemoryManager;
using hshm::ipc::Pointer;

extern Allocator *alloc_g;

GLOBAL_CONST allocator_id_t MAIN_ALLOC_ID(0, 1);

template<typename AllocT>
void Pretest() {
  std::string shm_url = "test_allocators";
  auto mem_mngr = HERMES_MEMORY_MANAGER;
  mem_mngr->UnregisterAllocator(MAIN_ALLOC_ID);
  mem_mngr->UnregisterBackend(shm_url);
  mem_mngr->CreateBackend<PosixShmMmap>(
    MEGABYTES(100), shm_url);
  mem_mngr->CreateAllocator<AllocT>(shm_url, MAIN_ALLOC_ID, 0);
  alloc_g = mem_mngr->GetAllocator(MAIN_ALLOC_ID);
}

void Posttest();

#endif  // HERMES_TEST_UNIT_DATA_STRUCTURES_TEST_INIT_H_
