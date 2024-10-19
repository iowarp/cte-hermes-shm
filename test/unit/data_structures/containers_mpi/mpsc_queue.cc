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
#include "hermes_shm/data_structures/ipc/string.h"
#include "hermes_shm/data_structures/ipc/mpsc_ptr_queue.h"
#include "hermes_shm/util/error.h"
#include "hermes_shm/util/affinity.h"

TEST_CASE("TestMpscQueueMpi") {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // The allocator was initialized in test_init.c
  // we are getting the "header" of the allocator
  Allocator *alloc = alloc_g;
  Pointer *header = alloc->GetCustomHeader<Pointer>();

  // Make the queue uptr
  hipc::uptr<hipc::mpsc_ptr_queue<int>> queue_;
  if (rank == 0) {
    // Rank 0 create the pointer queue
    queue_ = hipc::make_uptr<hipc::mpsc_ptr_queue<int>>(alloc, 256);
    queue_ >> (*header);
    // Affine to CPU 0
    ProcessAffiner::SetCpuAffinity(getpid(), 0);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank != 0) {
    // Rank 1 gets the pointer queue
    queue_ << (*header);
    // Affine to CPU 1
    ProcessAffiner::SetCpuAffinity(getpid(), 1);
  }

  if (rank == 0) {
    // Emplace values into the queue
    for (int i = 0; i < 256; ++i) {
      queue_->emplace(i);
    }
  } else {
    // Pop entries from the queue
    int x, count = 0;
    while (!queue_->pop(x).IsNull() && count < 256) {
      REQUIRE(x == count);
      ++count;
    }
  }

  // The barrier is necessary so that
  // Rank 0 doesn't exit before Rank 1
  // The uptr frees data when rank 0 exits.
  MPI_Barrier(MPI_COMM_WORLD);
}

