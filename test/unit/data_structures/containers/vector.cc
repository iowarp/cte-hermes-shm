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
#include "hermes_shm/data_structures/ipc/vector.h"
#include "hermes_shm/data_structures/ipc/list.h"
#include "hermes_shm/data_structures/ipc/string.h"
#include "vector.h"

using hshm::ipc::vector_templ;
using hshm::ipc::list;
using hshm::ipc::string;

template<typename T, bool FIXED>
void VectorTestRunner(VectorTestSuite<T, vector_templ<T, FIXED>> &test) {
  test.EmplaceTest(15);
  test.IndexTest();
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

template<typename T, bool FIXED, bool ptr>
void VectorTest() {
  Allocator *alloc = alloc_g;
  auto vec = hipc::make_uptr<vector_templ<T, FIXED>>(alloc);
  VectorTestSuite<T, vector_templ<T, FIXED>> test(*vec, alloc);
  VectorTestRunner<T>(test);
}

template<bool FIXED>
void VectorOfVectorOfStringTest() {
  Allocator *alloc = alloc_g;
  auto vec = hipc::make_uptr<
    vector_templ<vector_templ<string, FIXED>, FIXED>>(alloc);

  vec->resize(10);
  for (hipc::Ref<vector_templ<string, FIXED>> bkt : *vec) {
    bkt->emplace_back("hello");
  }
  vec->clear();
}

template<bool FIXED>
void VectorOfListOfStringTest() {
  Allocator *alloc = alloc_g;
  auto vec = hipc::make_uptr<vector_templ<list<string>, FIXED>>(alloc);

  vec->resize(10);

  PAGE_DIVIDE("Emplace an element into each bucket") {
    size_t count = 0;
    for (hipc::Ref<list<string>> bkt : *vec) {
      bkt->emplace_back(std::to_string(count));
      count += 1;
    }
    REQUIRE(count == 10);
  }

  PAGE_DIVIDE("Get string from each bucket") {
    size_t count = 0;
    for (hipc::Ref<list<string>> bkt : *vec) {
      for (hipc::Ref<string> val : *bkt) {
        REQUIRE(*val == std::to_string(count));
      }
      count += 1;
    }
    REQUIRE(count == 10);
  }

  vec->clear();
}

TEST_CASE("VectorOfInt") {
  Allocator *alloc = alloc_g;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  VectorTest<int, false, false>();
  VectorTest<int, false, true>();
  VectorTest<int, true, false>();
  VectorTest<int, true, true>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("VectorOfString") {
  Allocator *alloc = alloc_g;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  VectorTest<hipc::string, false, false>();
  VectorTest<int, false, true>();
  VectorTest<hipc::string, true, false>();
  VectorTest<int, true, true>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("VectorOfStdString") {
  Allocator *alloc = alloc_g;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  VectorTest<std::string, false, false>();
  VectorTest<int, false, true>();
  VectorTest<std::string, true, false>();
  VectorTest<int, true, true>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("VectorOfVectorOfString") {
  Allocator *alloc = alloc_g;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  VectorOfVectorOfStringTest<false>();
  VectorOfVectorOfStringTest<true>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("VectorOfListOfString") {
  Allocator *alloc = alloc_g;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  VectorOfListOfStringTest<false>();
  VectorOfListOfStringTest<true>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}
