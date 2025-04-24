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
#include "hermes_shm/data_structures/ipc/ring_ptr_queue.h"
#include "hermes_shm/data_structures/ipc/string.h"
#include "hermes_shm/util/affinity.h"
#include "hermes_shm/util/error.h"
#include "test_init.h"

TEST_CASE("TestMpscQueueMpi") {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // The allocator was initialized in test_init.c
  // we are getting the "header" of the allocator
  auto *alloc = HSHM_MEMORY_MANAGER->GetAllocator<HSHM_DEFAULT_ALLOC_T>(
      AllocatorId(1, 0));
  auto *queue_ =
      alloc->GetCustomHeader<hipc::delay_ar<sub::ipc::mpsc_ptr_queue<int>>>();

  // Make the queue uptr
  if (rank == RANK0) {
    // Rank 0 create the pointer queue
    queue_->shm_init(alloc, 256);
    // Affine to CPU 0
    hshm::ProcessAffiner::SetCpuAffinity(HSHM_SYSTEM_INFO->pid_, 0);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank != RANK0) {
    // Affine to CPU 1
    hshm::ProcessAffiner::SetCpuAffinity(HSHM_SYSTEM_INFO->pid_, 1);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  sub::ipc::mpsc_ptr_queue<int> *queue = queue_->get();
  if (rank == RANK0) {
    // Emplace values into the queue
    for (int i = 0; i < 256; ++i) {
      queue->emplace(i);
    }
  } else {
    // Pop entries from the queue
    int x, count = 0;
    while (!queue->pop(x).IsNull() && count < 256) {
      REQUIRE(x == count);
      ++count;
    }
  }

  // The barrier is necessary so that
  // Rank 0 doesn't exit before Rank 1
  // The uptr frees data when rank 0 exits.
  MPI_Barrier(MPI_COMM_WORLD);
}
