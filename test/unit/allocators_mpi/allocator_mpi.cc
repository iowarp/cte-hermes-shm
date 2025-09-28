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

#include <mpi.h>

#include "basic_test.h"
#include "test_init.h"

struct Record {
  char *data;
  size_t size;
  Pointer ptr;
};

template <typename AllocT>
void MpiPageAllocationTest(AllocT *alloc, size_t count) {
  size_t window_length = 32;
  size_t min_page = 64;
  size_t max_page = hshm::Unit<size_t>::Megabytes(1);
  std::mt19937 rng(23522523);
  std::uniform_int_distribution<size_t> uni(min_page, max_page);

  MPI_Barrier(MPI_COMM_WORLD);
  size_t num_windows = count / window_length;
  std::vector<Record> window(window_length);
  for (size_t w = 0; w < num_windows; ++w) {
    for (size_t i = 0; i < window_length; ++i) {
      window[i].size = uni(rng);
      auto full_ptr = alloc->Allocate(HSHM_DEFAULT_MEM_CTX, window[i].size);
      window[i].ptr = full_ptr.shm_;
      window[i].data = reinterpret_cast<char*>(full_ptr.ptr_);
      memset(window[i].data, (char)i, window[i].size);
    }
    for (size_t i = 0; i < window_length; ++i) {
      VerifyBuffer(window[i].data, window[i].size, (char)i);
      hipc::FullPtr<void> full_ptr(window[i].data, window[i].ptr);
      alloc->Free(HSHM_DEFAULT_MEM_CTX, full_ptr);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

template <typename AllocT>
AllocT *TestAllocatorMpi() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    PretestRank0<AllocT>();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank != 0) {
    PretestRankN();
  }
  return HSHM_MEMORY_MANAGER->GetAllocator<AllocT>(alloc_id);
}

TEST_CASE("StackAllocatorMpi") {
  auto alloc = TestAllocatorMpi<hipc::StackAllocator>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  MpiPageAllocationTest(alloc, 100);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("ScalablePageAllocatorMpi") {
  auto alloc = TestAllocatorMpi<hipc::ScalablePageAllocator>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  MpiPageAllocationTest(alloc, 1000);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}
