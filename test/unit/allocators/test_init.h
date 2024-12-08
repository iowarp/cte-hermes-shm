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


#ifndef HERMES_TEST_UNIT_ALLOCATORS_TEST_INIT_H_
#define HERMES_TEST_UNIT_ALLOCATORS_TEST_INIT_H_

#include "basic_test.h"
#include "omp.h"
#include "hermes_shm/memory/memory_manager.h"

using hshm::ipc::MemoryBackendType;
using hshm::ipc::MemoryBackend;
using hshm::ipc::AllocatorId;
using hshm::ipc::AllocatorType;
using hshm::ipc::Allocator;
using hshm::ipc::MemoryManager;
using hshm::ipc::Pointer;

#define HEADER_CHECKSUM 8482942

struct SimpleAllocatorHeader {
  int checksum_;
};

template<typename BackendT, typename AllocT>
Allocator* Pretest() {
  std::string shm_url = "test_allocators";
  AllocatorId alloc_id(0, 1);
  auto mem_mngr = HERMES_MEMORY_MANAGER;
  mem_mngr->UnregisterAllocator(alloc_id);
  mem_mngr->UnregisterBackend(hipc::MemoryBackendId::Get(0));
  mem_mngr->CreateBackendWithUrl<BackendT>(
      hipc::MemoryBackendId::Get(0), GIGABYTES(1), shm_url);
  mem_mngr->CreateAllocator<AllocT>(
      hipc::MemoryBackendId::Get(0), alloc_id, sizeof(SimpleAllocatorHeader));
  auto alloc = mem_mngr->GetAllocator<AllocT>(alloc_id);
  auto hdr = alloc->template GetCustomHeader<SimpleAllocatorHeader>();
  hdr->checksum_ = HEADER_CHECKSUM;
  return alloc;
}

void Posttest();
void PageAllocationTest(Allocator *alloc);
void MultiPageAllocationTest(Allocator *alloc);

#endif  // HERMES_TEST_UNIT_ALLOCATORS_TEST_INIT_H_
