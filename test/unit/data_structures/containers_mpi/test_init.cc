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

#include "basic_test.h"

template <typename AllocT>
void PretestRank0() {
  std::string shm_url = "test_allocators2";
  AllocatorId alloc_id(1, 0);
  auto mem_mngr = HSHM_MEMORY_MANAGER;
  mem_mngr->UnregisterAllocator(alloc_id);
  mem_mngr->DestroyBackend(hipc::MemoryBackendId::GetRoot());
  mem_mngr->CreateBackend<PosixShmMmap>(hipc::MemoryBackendId::Get(0),
                                        hshm::Unit<size_t>::Megabytes(100),
                                        shm_url);
  mem_mngr->CreateAllocator<AllocT>(hipc::MemoryBackendId::Get(0), alloc_id,
                                    sizeof(sub::ipc::mpsc_ptr_queue<int>));
}

void PretestRankN() {
  std::string shm_url = "test_allocators2";
  AllocatorId alloc_id(1, 0);
  auto mem_mngr = HSHM_MEMORY_MANAGER;
  mem_mngr->AttachBackend(MemoryBackendType::kPosixShmMmap, shm_url);
}

void MainPretest() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  HILOG(kInfo, "PRETEST RANK 0 beginning {}", rank);
  if (rank == RANK0) {
    PretestRank0<HSHM_DEFAULT_ALLOC_T>();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank != RANK0) {
    PretestRankN();
  }
  HILOG(kInfo, "PRETEST RANK 0 done {}", rank);
  MPI_Barrier(MPI_COMM_WORLD);
}

void MainPosttest() {}
