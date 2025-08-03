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

#include "queue.h"

#include "basic_test.h"
#include "hermes_shm/data_structures/ipc/multi_ring_buffer.h"
#include "test_init.h"

/**
 * TEST TICKET QUEUE
 * */

TEST_CASE("TestTicketQueueInt") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  ProduceThenConsume<hipc::ticket_queue<int>, int>(1, 1, 32, 32);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("TestTicketQueueIntMultiThreaded") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  ProduceAndConsume<hipc::ticket_queue<int>, int>(8, 1, 8192, 64);
  ProduceAndConsume<hipc::ticket_queue<int>, int>(8, 8, 8192, 64);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("TestSplitTicketQueueInt") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  ProduceThenConsume<hipc::split_ticket_queue<int>, int>(1, 1, 32, 32);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("TestSplitTicketQueueIntMultiThreaded") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  ProduceAndConsume<hipc::split_ticket_queue<int>, int>(8, 1, 8192, 64);
  ProduceAndConsume<hipc::split_ticket_queue<int>, int>(8, 8, 8192, 64);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

/**
 * TEST DYNAMIC QUEUE
 * */

TEST_CASE("TestDynamicQueueInt") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  ProduceThenConsume<hipc::dynamic_queue<int>, int>(1, 1, 32, 32);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("TestDynamicQueueString") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  ProduceThenConsume<hipc::dynamic_queue<hipc::string>, hipc::string>(1, 1, 32,
                                                                      32);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("TestDynamicQueueIntMultiThreaded") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  ProduceAndConsume<hipc::dynamic_queue<int>, int>(8, 1, 8192, 32);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("TestDynamicQueueStringMultiThreaded") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  ProduceAndConsume<hipc::dynamic_queue<hipc::string>, hipc::string>(8, 1, 8192,
                                                                     32);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

/**
 * TEST SPSC LIST QUEUE
 * */

TEST_CASE("TestSpscListQueueInt") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  ProduceThenConsume<hipc::spsc_fifo_list_queue<IntEntry>, IntEntry *>(1, 1, 32,
                                                                       32);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

/**
 * TEST MPSC LIFO LIST QUEUE
 * */

TEST_CASE("TestMpscLifoListQueueInt") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  ProduceThenConsume<hipc::mpsc_lifo_list_queue<IntEntry>, IntEntry *>(1, 1, 32,
                                                                       32);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("TestMpscLifoListQueueIntMultithreaded") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  ProduceAndConsume<hipc::mpsc_lifo_list_queue<IntEntry>, IntEntry *>(
      8, 1, 48000, 32);
  ProduceAndConsume<hipc::mpsc_lifo_list_queue<IntEntry>, IntEntry *>(
      8, 1, 48000, 32);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

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
  ProduceThenConsume<hipc::mpsc_queue<hipc::string>, hipc::string>(1, 1, 32,
                                                                   32);
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
  ProduceAndConsume<hipc::mpsc_queue<hipc::string>, hipc::string>(8, 1, 8192,
                                                                  32);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("TestMpscQueuePeek") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  auto q = alloc->NewObjLocal<hipc::mpsc_queue<int>>(HSHM_DEFAULT_MEM_CTX);
  q->emplace(1);
  int *val;
  q->peek(val, 0);
  REQUIRE(*val == 1);
  hipc::pair<hshm::bitfield64_t, int> *val_pair;
  q->peek(val_pair, 0);
  REQUIRE(val_pair->GetSecond() == 1);
  alloc->DelObjLocal(HSHM_DEFAULT_MEM_CTX, q);

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

/**
 * TEST SPSC QUEUE
 * */

TEST_CASE("TestSpscQueueInt") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  ProduceThenConsume<hipc::spsc_queue<int>, int>(1, 1, 32, 32);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("TestSpscQueueString") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  ProduceThenConsume<hipc::spsc_queue<hipc::string>, hipc::string>(1, 1, 32,
                                                                   32);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("TestSpscQueuePopBack") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  PAGE_DIVIDE("TEST") {
    hshm::spsc_queue<int> queue(alloc);
    queue.emplace(1);
    queue.emplace(2);
    queue.emplace(3);
    int val;
    queue.pop_back(val);
    REQUIRE(val == 3);
    queue.pop_back(val);
    REQUIRE(val == 2);
    queue.pop_back(val);
    REQUIRE(val == 1);
    REQUIRE(queue.pop_back(val).IsNull());
  }
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

template <typename T>
void PointerQueueTest(T base_val) {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  hshm::spsc_ptr_queue<T> queue(alloc);
  queue.emplace(base_val + 1);
  queue.emplace(base_val + 2);
  queue.emplace(base_val + 3);
  T val;
  queue.pop_back(val);
  REQUIRE(val == base_val + 3);
  queue.pop_back(val);
  REQUIRE(val == base_val + 2);
  queue.pop_back(val);
  REQUIRE(val == base_val + 1);
  REQUIRE(queue.pop_back(val).IsNull());
}

TEST_CASE("TestSpscPtrQueuePopBack") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  PointerQueueTest<int>(0);
  PointerQueueTest<size_t>(0);
  PointerQueueTest<hipc::Pointer>(hipc::Pointer(alloc->id_, 0));
  PointerQueueTest<hipc::OffsetPointer>(hipc::OffsetPointer(0));
  PointerQueueTest<hipc::FullPtr<char>>(
      hipc::FullPtr<char>(nullptr, hipc::Pointer(alloc->id_, 0)));
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

/**
 * TEST MULTI RING BUFFER
 * */

TEST_CASE("TestMultiRingBufferBasic") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("TEST") {
    // Create a multi-ring buffer with 2 lanes, 3 priorities, depth 8
    hshm::multi_mpsc_queue<int> buffer(alloc, 2, 3, 8);

    // Test basic properties
    REQUIRE(buffer.GetNumLanes() == 2);
    REQUIRE(buffer.GetNumPriorities() == 3);

    // Push items to different lanes and priorities using GetLane
    auto &lane_0_pri_0 = buffer.GetLane(0, 0);  // Lane 0, Priority 0 (highest)
    auto &lane_0_pri_1 = buffer.GetLane(0, 1);  // Lane 0, Priority 1
    auto &lane_1_pri_0 = buffer.GetLane(1, 0);  // Lane 1, Priority 0 (highest)
    auto &lane_1_pri_2 = buffer.GetLane(1, 2);  // Lane 1, Priority 2 (lowest)

    auto tok1 = lane_0_pri_0.emplace(100);
    auto tok2 = lane_0_pri_1.emplace(200);
    auto tok3 = lane_1_pri_0.emplace(300);
    auto tok4 = lane_1_pri_2.emplace(400);

    REQUIRE(!tok1.IsNull());
    REQUIRE(!tok2.IsNull());
    REQUIRE(!tok3.IsNull());
    REQUIRE(!tok4.IsNull());

    // Check sizes
    REQUIRE(lane_0_pri_0.GetSize() == 1);
    REQUIRE(lane_0_pri_1.GetSize() == 1);
    REQUIRE(lane_1_pri_0.GetSize() == 1);
    REQUIRE(lane_1_pri_2.GetSize() == 1);

    // Pop from individual lanes
    int val;
    auto pop_tok = lane_0_pri_0.pop(val);
    REQUIRE(!pop_tok.IsNull());
    REQUIRE(val == 100);

    pop_tok = lane_0_pri_1.pop(val);
    REQUIRE(!pop_tok.IsNull());
    REQUIRE(val == 200);

    pop_tok = lane_1_pri_0.pop(val);
    REQUIRE(!pop_tok.IsNull());
    REQUIRE(val == 300);

    pop_tok = lane_1_pri_2.pop(val);
    REQUIRE(!pop_tok.IsNull());
    REQUIRE(val == 400);

    // All items popped

    // Try to pop from empty lanes
    pop_tok = lane_0_pri_0.pop(val);
    REQUIRE(pop_tok.IsNull());
  }

  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("TestMultiRingBufferLaneAccess") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("TEST") {
    // Create a multi-ring buffer with 3 lanes, 2 priorities
    hshm::multi_mpsc_queue<int> buffer(alloc, 3, 2, 8);

    // Access different lanes and priorities
    auto &lane_0_pri_0 = buffer.GetLane(0, 0);
    auto &lane_1_pri_0 = buffer.GetLane(1, 0);
    auto &lane_2_pri_0 = buffer.GetLane(2, 0);
    auto &lane_0_pri_1 = buffer.GetLane(0, 1);

    // Push to different lanes
    lane_0_pri_0.emplace(1);
    lane_1_pri_0.emplace(2);
    lane_2_pri_0.emplace(3);
    lane_0_pri_1.emplace(4);

    REQUIRE(lane_0_pri_0.GetSize() == 1);  // Lane 0, Priority 0: value 1
    REQUIRE(lane_1_pri_0.GetSize() == 1);  // Lane 1, Priority 0: value 2
    REQUIRE(lane_2_pri_0.GetSize() == 1);  // Lane 2, Priority 0: value 3
    REQUIRE(lane_0_pri_1.GetSize() == 1);  // Lane 0, Priority 1: value 4

    // Pop from individual lanes
    int val;
    auto tok1 = lane_0_pri_0.pop(val);
    REQUIRE(!tok1.IsNull());
    REQUIRE(val == 1);

    auto tok2 = lane_1_pri_0.pop(val);
    REQUIRE(!tok2.IsNull());
    REQUIRE(val == 2);

    auto tok3 = lane_2_pri_0.pop(val);
    REQUIRE(!tok3.IsNull());
    REQUIRE(val == 3);

    auto tok4 = lane_0_pri_1.pop(val);
    REQUIRE(!tok4.IsNull());
    REQUIRE(val == 4);

  }

  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("TestMultiRingBufferBoundsChecking") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("TEST") {
    // Create a multi-ring buffer with 2 lanes, 2 priorities
    hshm::multi_mpsc_queue<int> buffer(alloc, 2, 2, 8);

    // Valid access should work
    auto &valid_lane = buffer.GetLane(1, 1);
    valid_lane.emplace(400);
    REQUIRE(valid_lane.GetSize() == 1);

    int val;
    auto tok = valid_lane.pop(val);
    REQUIRE(!tok.IsNull());
    REQUIRE(val == 400);
  }

  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("TestMultiRingBufferResize") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("TEST") {
    // Create a small multi-ring buffer
    hshm::multi_mpsc_queue<int> buffer(alloc, 2, 2, 4);

    // Fill it up using lane access
    auto &lane_0_0 = buffer.GetLane(0, 0);
    auto &lane_0_1 = buffer.GetLane(0, 1);
    auto &lane_1_0 = buffer.GetLane(1, 0);
    auto &lane_1_1 = buffer.GetLane(1, 1);

    for (int i = 0; i < 2; ++i) {  // Fill each lane with 2 items
      lane_0_0.emplace(i);
      lane_0_1.emplace(i + 10);
      lane_1_0.emplace(i + 20);
      lane_1_1.emplace(i + 30);
    }

  }

  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("TestMultiRingBufferDirectLaneAccess") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("TEST") {
    // Create a multi-ring buffer
    hshm::multi_mpsc_queue<int> buffer(alloc, 2, 2, 8);

    // Get direct access to a specific lane
    auto &lane_0_1 = buffer.GetLane(0, 1);  // Lane 0, Priority 1

    // Use the lane directly
    lane_0_1.emplace(42);
    REQUIRE(lane_0_1.GetSize() == 1);
    REQUIRE(lane_0_1.GetSize() == 1);

    // Pop using direct access
    int val;
    auto tok = lane_0_1.pop(val);
    REQUIRE(!tok.IsNull());
    REQUIRE(val == 42);
    REQUIRE(lane_0_1.GetSize() == 0);

    // Test const access
    const auto &const_buffer = buffer;
    const auto &const_lane = const_buffer.GetLane(1, 0);
    // Note: GetSize() is not const, so we can't test it on const objects
  }

  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}
