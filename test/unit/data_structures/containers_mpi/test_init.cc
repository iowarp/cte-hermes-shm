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


#include "basic_test.h"
#include "test_init.h"

template<typename AllocT>
void PretestRank0() {
  std::string shm_url = "test_allocators";
  AllocatorId alloc_id(0, 1);
  auto mem_mngr = HERMES_MEMORY_MANAGER;
  mem_mngr->UnregisterAllocator(alloc_id);
  mem_mngr->UnregisterBackend(hipc::MemoryBackendId::GetRoot());
  mem_mngr->CreateBackend<PosixShmMmap>(
      hipc::MemoryBackendId::Get(0), MEGABYTES(100), shm_url);
  mem_mngr->CreateAllocator<AllocT>(
      hipc::MemoryBackendId::Get(0), alloc_id, 0);
}

void PretestRankN() {
  std::string shm_url = "test_allocators";
  AllocatorId alloc_id(0, 1);
  auto mem_mngr = HERMES_MEMORY_MANAGER;
  mem_mngr->AttachBackend(MemoryBackendType::kPosixShmMmap, shm_url);
}

void MainPretest() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    // PretestRank0<hipc::StackAllocator>();
    PretestRank0<hipc::ScalablePageAllocator>();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank != 0) {
    PretestRankN();
  }
}

void MainPosttest() {
}
