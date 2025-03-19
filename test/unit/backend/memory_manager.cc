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

#include "hermes_shm/memory/memory_manager.h"

#include <mpi.h>

#include "basic_test.h"

using hshm::ipc::AllocatorId;
using hshm::ipc::AllocatorType;
using hshm::ipc::MemoryBackend;
using hshm::ipc::MemoryBackendType;
using hshm::ipc::MemoryManager;

struct SimpleHeader {
  hshm::ipc::Pointer p_;
};

TEST_CASE("MemoryManager") {
  int rank;
  char nonce = 8;
  size_t page_size = hshm::Unit<size_t>::Kilobytes(4);
  std::string shm_url = "test_mem_backend";
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  AllocatorId alloc_id(1, 0);

  HSHM_ERROR_HANDLE_START()
  auto mem_mngr = HSHM_MEMORY_MANAGER;

  if (rank == 0) {
    std::cout << "Creating SHMEM (rank 0): " << shm_url << std::endl;
    mem_mngr->UnregisterAllocator(alloc_id);
    mem_mngr->DestroyBackend(hipc::MemoryBackendId::Get(0));
    mem_mngr->CreateBackend<hipc::PosixShmMmap>(
        hipc::MemoryBackendId::Get(0), hshm::Unit<size_t>::Megabytes(100),
        shm_url);
    mem_mngr->CreateAllocator<HSHM_DEFAULT_ALLOC_T>(
        hipc::MemoryBackendId::Get(0), alloc_id, 0);
    mem_mngr->ScanBackends();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank != 0) {
    std::cout << "Attaching SHMEM (rank 1): " << shm_url << std::endl;
    mem_mngr->AttachBackend(MemoryBackendType::kPosixShmMmap, shm_url);
    mem_mngr->ScanBackends();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout << "Allocating pages (rank 0)" << std::endl;
    auto *alloc = mem_mngr->GetAllocator<HSHM_DEFAULT_ALLOC_T>(alloc_id);
    char *page =
        alloc->template AllocatePtr<char>(HSHM_DEFAULT_MEM_CTX, page_size);
    memset(page, nonce, page_size);
    auto header = alloc->GetCustomHeader<SimpleHeader>();
    hipc::Pointer p1 = mem_mngr->Convert<void>(alloc_id, page);
    hipc::Pointer p2 = mem_mngr->Convert<char>(page);
    header->p_ = p1;
    REQUIRE(p1 == p2);
    REQUIRE(VerifyBuffer(page, page_size, nonce));
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank != 0) {
    std::cout << "Finding and checking pages (rank 1)" << std::endl;
    auto *alloc = mem_mngr->GetAllocator<HSHM_DEFAULT_ALLOC_T>(alloc_id);
    SimpleHeader *header = alloc->template GetCustomHeader<SimpleHeader>();
    char *page = alloc->Convert<char>(header->p_);
    REQUIRE(VerifyBuffer(page, page_size, nonce));
  }
  MPI_Barrier(MPI_COMM_WORLD);

  HSHM_ERROR_HANDLE_END()
}
