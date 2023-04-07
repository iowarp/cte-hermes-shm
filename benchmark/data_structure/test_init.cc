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

#include "test_init.h"
#include "hermes_shm/data_structures/ipc/vector.h"
#include "hermes_shm/memory/allocator/stack_allocator.h"

std::unique_ptr<bipc::void_allocator> alloc_inst_g;
std::unique_ptr<bipc::managed_shared_memory> segment_g;

void MainPretest() {
  // Boost shared memory
  bipc::shared_memory_object::remove("HermesBoostBench");
  segment_g = std::make_unique<bipc::managed_shared_memory>(
    bipc::create_only, "HermesBoostBench", GIGABYTES(4));
  alloc_inst_g = std::make_unique<bipc::void_allocator>(
    segment_g->get_segment_manager());

  // hermes shared memory
  std::string shm_url = "HermesBench";
  allocator_id_t alloc_id(0, 1);
  auto mem_mngr = HERMES_MEMORY_MANAGER;
  mem_mngr->CreateBackend<hipc::PosixShmMmap>(
    MemoryManager::kDefaultBackendSize, shm_url);
}

void MainPosttest() {
  bipc::shared_memory_object::remove("LabstorBoostBench");
}
