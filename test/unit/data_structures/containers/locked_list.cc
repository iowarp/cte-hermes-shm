//
// Created by lukemartinlogan on 1/21/23.
//

#include "basic_test.h"
#include "test_init.h"
#include "test_init.h"
#include "hermes_shm/data_structures/string.h"
#include <hermes_shm/data_structures/thread_safe/list.h>
#include <hermes_shm/data_structures/thread_unsafe/list.h>
#include "list.h"

template<typename T>
using lock_list = hermes::ipc::lock::list<T>;
using hermes::ipc::list;

template<typename T>
void LockedListTest() {
  Allocator *alloc = alloc_g;
  lock_list<T> lp(alloc);
  ListTestSuite<T, list<T>> test(lp.GetContainer(), alloc);

  test.EmplaceTest(30);
  test.ForwardIteratorTest();
  test.ConstForwardIteratorTest();
  test.CopyConstructorTest();
  test.CopyAssignmentTest();
  test.MoveConstructorTest();
  /*test.MoveAssignmentTest();
  test.EmplaceFrontTest();
  test.ModifyEntryCopyIntoTest();
  test.ModifyEntryMoveIntoTest();
  test.EraseTest();*/
}

TEST_CASE("LockedListOfInt") {
  Allocator *alloc = alloc_g;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  LockedListTest<int>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("LockedListOfString") {
  Allocator *alloc = alloc_g;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  LockedListTest<hipc::string>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("LockedListOfStdString") {
  Allocator *alloc = alloc_g;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  LockedListTest<std::string>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}