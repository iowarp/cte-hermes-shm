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
#include "hermes_shm/data_structures/ipc/lifo_list_queue.h"
#include "lifo_list_queue.h"
#include "test_init.h"

using hshm::ipc::lifo_list_queue;

template <typename T>
void lifo_list_queueTest() {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  lifo_list_queue<T> lp(alloc);
  lifo_list_queueTestSuite<T, lifo_list_queue<T>> test(lp, alloc);

  test.EnqueueTest(30);
  test.ForwardIteratorTest();
  test.ConstForwardIteratorTest();
  test.DequeueTest(30);
  test.DequeueMiddleTest();
  test.EraseTest();
}

TEST_CASE("lifo_list_queueOfMpPage") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  lifo_list_queueTest<MpPage>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}
