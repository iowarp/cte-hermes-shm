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

#include <iostream>
#include <memory>

#include "hermes_shm/data_structures/ipc/string.h"

void MainPretest() {
  std::string shm_url = "test_serializers";
  AllocatorId alloc_id(1, 0);
  auto mem_mngr = HSHM_MEMORY_MANAGER;
  mem_mngr->UnregisterAllocator(alloc_id);
  mem_mngr->DestroyBackend(hipc::MemoryBackendId::Get(0));
  mem_mngr->CreateBackend<PosixShmMmap>(hipc::MemoryBackendId::Get(0),
                                        hshm::Unit<size_t>::Megabytes(100),
                                        shm_url);
  mem_mngr->CreateAllocator<HSHM_DEFAULT_ALLOC_T>(hipc::MemoryBackendId::Get(0),
                                                  alloc_id, 0);
}

void MainPosttest() {}
