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

#include "hermes_shm/data_structures/ipc/vector.h"

#include "basic_test.h"
#include "hermes_shm/data_structures/ipc/list.h"
#include "hermes_shm/data_structures/ipc/string.h"
#include "test_init.h"
#include "vector.h"

using hshm::ipc::list;
using hshm::ipc::string;
using hshm::ipc::vector;

template <typename T>
void VectorTestRunner(VectorTestSuite<T, vector<T>> &test) {
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

template <typename T, bool ptr>
void VectorTest() {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  auto vec = vector<T>(alloc);
  VectorTestSuite<T, vector<T>> test(vec, alloc);
  VectorTestRunner<T>(test);
}

void VectorOfVectorOfStringTest() {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  auto vec = vector<vector<string>>(alloc);

  vec.resize(10);
  for (vector<string> &bkt : vec) {
    bkt.emplace_back("hello");
  }
  vec.clear();
}

void VectorOfListOfStringTest() {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  auto vec = vector<list<string>>(alloc);

  vec.resize(10);

  PAGE_DIVIDE("Emplace an element into each bucket") {
    size_t count = 0;
    for (list<string> &bkt : vec) {
      bkt.emplace_back(std::to_string(count));
      count += 1;
    }
    REQUIRE(count == 10);
  }

  PAGE_DIVIDE("Get string from each bucket") {
    size_t count = 0;
    for (list<string> &bkt : vec) {
      for (string &val : bkt) {
        REQUIRE(val == std::to_string(count));
      }
      count += 1;
    }
    REQUIRE(count == 10);
  }

  vec.clear();
}

TEST_CASE("VectorOfInt") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  try {
    REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
    VectorTest<int, false>();
    VectorTest<int, true>();
    REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  } catch (hshm::Error &e) {
    FAIL(e.what());
  } catch (std::exception &e) {
    FAIL(e.what());
  }
}

TEST_CASE("VectorOfString") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  VectorTest<hipc::string, false>();
  VectorTest<int, true>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("VectorOfStdString") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  VectorTest<std::string, false>();
  VectorTest<int, true>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("VectorOfVectorOfString") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  VectorOfVectorOfStringTest();
  VectorOfVectorOfStringTest();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("VectorOfListOfString") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  VectorOfListOfStringTest();
  VectorOfListOfStringTest();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("VectorOfIntInsertionSort") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  hshm::vector<int> vec(alloc);
  vec.resize(10);
  for (int i = 0; i < 10; ++i) {
    vec[i] = 10 - i;
  }
  hshm::insertion_sort(vec.begin(), vec.end());
  REQUIRE(hshm::is_sorted(vec.begin(), vec.end()));
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("VectorOfIntQuicksort") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  hshm::vector<int> vec(alloc);
  int length = 91;
  vec.resize(length);
  for (int i = 0; i < length; ++i) {
    vec[i] = length - i;
  }
  hshm::quick_sort(vec.begin(), vec.end());
  REQUIRE(hshm::is_sorted(vec.begin(), vec.end()));
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("VectorOfIntHeapSort") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  hshm::vector<int> vec(alloc);
  int length = 91;
  vec.resize(length);
  for (int i = 0; i < length; ++i) {
    vec[i] = length - i;
  }
  hshm::heap_sort(vec.begin(), vec.end());
  REQUIRE(hshm::is_sorted(vec.begin(), vec.end()));
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("VectorOfIntQuickSortLambda") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  hshm::vector<int> vec(alloc);
  int length = 91;
  vec.resize(length);
  for (int i = 0; i < length; ++i) {
    vec[i] = length - i;
  }
  hshm::quick_sort(vec.begin(), vec.end(),
                   [](const int &a, const int &b) { return a < b; });
  REQUIRE(hshm::is_sorted(vec.begin(), vec.end()));
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}