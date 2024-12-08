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
#include "hermes_shm/data_structures/ipc/ring_queue.h"
#include "hermes_shm/data_structures/ipc/ring_ptr_queue.h"
#include "queue.h"

/**
 * TEST MPSC QUEUE
 * */

TEST_CASE("TestMpscQueueInt") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  ProduceThenConsume<hipc::mpsc_queue<int>, int>(1, 1, 32, 32);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("TestMpscQueueString") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  ProduceThenConsume<hipc::mpsc_queue<hipc::string>, hipc::string>(
    1, 1, 32, 32);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("TestMpscQueueIntMultiThreaded") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  ProduceAndConsume<hipc::mpsc_queue<int>, int>(8, 1, 8192, 32);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("TestMpscQueueStringMultiThreaded") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  ProduceAndConsume<hipc::mpsc_queue<hipc::string>, hipc::string>(
    8, 1, 8192, 32);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("TestMpscQueuePeek") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  auto q = alloc->NewObjLocal<hipc::mpsc_queue<int>>(HSHM_DEFAULT_MEM_CTX).ptr_;
  q->emplace(1);
  int *val;
  q->peek(val, 0);
  REQUIRE(*val == 1);
  hipc::pair<hshm::bitfield32_t, int> *val_pair;
  q->peek(val_pair, 0);
  REQUIRE(val_pair->GetSecond() == 1);
  alloc->DelObj(HSHM_DEFAULT_MEM_CTX, q);

  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

/**
 * MPSC Pointer Queue
 * */

TEST_CASE("TestMpscPtrQueueInt") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  ProduceThenConsume<hipc::mpsc_ptr_queue<int>, int>(1, 1, 32, 32);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("TestMpscPtrQueueIntMultiThreaded") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  ProduceAndConsume<hipc::mpsc_ptr_queue<int>, int>(8, 1, 8192, 32);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("TestMpscOffsetPointerQueueCompile") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  hipc::mpsc_ptr_queue<hipc::OffsetPointer> queue(alloc);
  hipc::OffsetPointer off_p;
  queue.emplace(hipc::OffsetPointer(5));
  queue.pop(off_p);
  REQUIRE(off_p == hipc::OffsetPointer(5));
}

TEST_CASE("TestMpscPointerQueueCompile") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  hipc::mpsc_ptr_queue<hipc::Pointer> queue(alloc);
  hipc::Pointer off_p;
  queue.emplace(hipc::Pointer(AllocatorId(5, 2), 1));
  queue.pop(off_p);
  REQUIRE(off_p == hipc::Pointer(AllocatorId(5, 2), 1));
}
