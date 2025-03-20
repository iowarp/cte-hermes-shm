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

#include "list.h"

#include "basic_test.h"
#include "hermes_shm/data_structures/ipc/list.h"
#include "hermes_shm/data_structures/ipc/string.h"
#include "test_init.h"

template <typename T, typename ListT>
void ListTestRunner(ListTestSuite<T, ListT> &test) {
  test.EmplaceTest(15);
  test.ForwardIteratorTest();
  test.ConstForwardIteratorTest();
  test.CopyConstructorTest();
  test.CopyAssignmentTest();
  test.MoveConstructorTest();
  test.MoveAssignmentTest();
  test.EmplaceFrontTest();
  test.ModifyEntryCopyIntoTest();
  test.ModifyEntryMoveIntoTest();
  test.EraseTest();
}

/**
 * HIPC list tests
 * */

template <typename T>
void HipcListTest() {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  hipc::list<T> lp(alloc);
  ListTestSuite<T, hipc::list<T>> test(lp, alloc);
  ListTestRunner(test);
}

TEST_CASE("hipc::ListOfInt") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  HipcListTest<int>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("hipc::ListOfString") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  HipcListTest<hipc::string>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("hipc::ListOfStdString") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  HipcListTest<std::string>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

/**
 * HSHM list tests
 * */

template <typename T>
void HshmListTest() {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  hshm::list<T> lp;
  ListTestSuite<T, hshm::list<T>> test(lp, alloc);
  ListTestRunner(test);
}
TEST_CASE("hshm::ListOfInt") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  HshmListTest<int>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}
TEST_CASE("hshm::ListOfString") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  HshmListTest<hipc::string>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}
TEST_CASE("hshm::ListOfStdString") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  HshmListTest<std::string>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}
