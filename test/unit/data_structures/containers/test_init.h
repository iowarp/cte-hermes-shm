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


#ifndef HERMES_TEST_UNIT_DATA_STRUCTURES_TEST_INIT_H_
#define HERMES_TEST_UNIT_DATA_STRUCTURES_TEST_INIT_H_

#include "hermes_shm/memory/allocator/page_allocator.h"
#include "hermes_shm/data_structures/data_structure.h"

using hermes::ipc::PosixShmMmap;
using hermes::ipc::MemoryBackendType;
using hermes::ipc::MemoryBackend;
using hermes::ipc::allocator_id_t;
using hermes::ipc::AllocatorType;
using hermes::ipc::Allocator;
using hermes::ipc::Pointer;

using hermes::ipc::MemoryBackendType;
using hermes::ipc::MemoryBackend;
using hermes::ipc::allocator_id_t;
using hermes::ipc::AllocatorType;
using hermes::ipc::Allocator;
using hermes::ipc::MemoryManager;
using hermes::ipc::Pointer;

extern Allocator *alloc_g;

template<typename AllocT>
void Pretest() {
  std::string shm_url = "test_allocators";
  allocator_id_t alloc_id(0, 1);
  auto mem_mngr = HERMES_MEMORY_MANAGER;
  mem_mngr->CreateBackend<PosixShmMmap>(
    MemoryManager::kDefaultBackendSize, shm_url);
  mem_mngr->CreateAllocator<AllocT>(shm_url, alloc_id, 0);
  alloc_g = mem_mngr->GetAllocator(alloc_id);
}

void Posttest();

#endif  // HERMES_TEST_UNIT_DATA_STRUCTURES_TEST_INIT_H_
